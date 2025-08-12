import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_rel
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import preprocess


df1 = preprocess.df1
final_classified_df = preprocess.final_classified_df
chemical_cols = preprocess.chemical_cols
inverse_clr_transform = preprocess.inverse_clr_transform

# =============================================================
# 2.1 风化与物理属性的统计关联分析 (Chi-squared Test)
# =============================================================
print("\n--- 问题1：风化关系分析与成分预测 ---\n")
print("2.1 风化与物理属性的关联分析:")

# 创建一个不含重复文物编号的数据集进行检验，避免样本不独立
# （例如，文物03有两个采样点，但其类型和纹饰是唯一的）
unique_artifacts_df = df1.drop_duplicates(subset=['文物编号'])

# 风化 vs. 玻璃类型
contingency_table_type = pd.crosstab(unique_artifacts_df['表面风化'], unique_artifacts_df['类型'])
chi2, p, dof, ex = chi2_contingency(contingency_table_type)
print(f"风化状态 vs. 玻璃类型 的卡方检验 p-value: {p:.4f}")
if p < 0.05:
    print("结论: 风化与玻璃类型存在显著关联。")
else:
    print("结论: 风化与玻璃类型无显著关联。")
print(contingency_table_type)

# 风化 vs. 纹饰
contingency_table_pattern = pd.crosstab(unique_artifacts_df['表面风化'], unique_artifacts_df['纹饰'])
chi2, p, dof, ex = chi2_contingency(contingency_table_pattern)
print(f"\n风化状态 vs. 纹饰 的卡方检验 p-value: {p:.4f}")
if p < 0.05:
    print("结论: 风化与纹饰存在显著关联。")
else:
    print("结论: 风化与纹饰无显著关联。")
print(contingency_table_pattern)


# =============================================================
# 2.2 风化化学特征量化 (Quantifying Chemical Effects)
# =============================================================
print("\n2.2 风化化学特征量化 (配对t检验):")

# 筛选成对样本 (风化 vs. 未风化)
paired_samples = []
# 示例：'49' (风化) vs '49未风化点' (无风化)
# '08' (风化) vs '08严重风化点' (严重风化) -> 这个比较风化程度，也可以用
# 我们需要找到同一文物编号下，风化状态不同的样本
# 手动识别配对样本以确保准确性
# (49, 49未风化点), (50, 50未风化点), (23, 23未风化点), (25, 25未风化点) ...
# (08, 08严重风化点), (26, 26严重风化点), (54, 54严重风化点)
# 为简化代码，我们手动定义这些对
pairs = {
    '49未风化点': '49', 
    '50未风化点': '50',
    '23未风化点': '23', # 假设23本身是风化样本
    '25未风化点': '25', # 假设25本身是风化样本
    '28未风化点': '28',
    '29未风化点': '29',
    '42未风化点1': '42',
    '44未风化点': '44',
    '53未风化点': '53',
    '08': '08严重风化点',
    '26': '26严重风化点',
    '54': '54严重风化点'
}

# 统一方向：仅保留“未风化” -> “风化/严重风化”的配对，确保差值方向恒为 (风化 - 未风化)
status_series = final_classified_df.set_index('文物采样点')['表面风化']
pairs_filtered = {}
for k, v in pairs.items():
    status_k = status_series.get(k, None)
    status_v = status_series.get(v, None)
    if status_k == '无风化' and status_v is not None and status_v != '无风化':
        pairs_filtered[k] = v

if len(pairs_filtered) == 0:
    print("警告：未找到方向一致（未风化→风化）的配对样本，使用原始配对但方向可能不一致。")
    pairs_effective = pairs
else:
    pairs_effective = pairs_filtered
    if len(pairs_effective) < len(pairs):
        print(f"已过滤不符合方向的配对 {len(pairs) - len(pairs_effective)} 对，保留 {len(pairs_effective)} 对用于统计。")

# 提取CLR变换后的数据
unweathered_paired = final_classified_df[final_classified_df['文物采样点'].isin(pairs_effective.keys())]
weathered_paired = final_classified_df[final_classified_df['文物采样点'].isin(pairs_effective.values())]

# 确保配对顺序正确
unweathered_paired = unweathered_paired.set_index('文物采样点')
weathered_paired = weathered_paired.set_index('文物采样点')
weathered_paired = weathered_paired.rename(index={v: k for k, v in pairs_effective.items()})
unweathered_paired = unweathered_paired.reindex(weathered_paired.index)

clr_cols = [f"CLR_{col}" for col in chemical_cols]
diff_clr = weathered_paired[clr_cols].values - unweathered_paired[clr_cols].values

print(f"已匹配配对样本数量: {len(weathered_paired)} 对")

# 对每个化学成分进行配对t检验，并输出方向（均值差）
results_rows = []
for i, col in enumerate(chemical_cols):
    w = weathered_paired[clr_cols[i]]
    u = unweathered_paired[clr_cols[i]]
    mean_delta = (w - u).mean()
    t_stat, p_val = ttest_rel(w, u)
    direction = '富集(风化↑)' if mean_delta > 0 else ('流失(风化↓)' if mean_delta < 0 else '无明显方向')
    results_rows.append({
        '成分': col,
        '均值差(风化-未风化, CLR)': mean_delta,
        't统计量': t_stat,
        'p值': p_val,
        '方向': direction,
        '显著性': '显著' if p_val < 0.05 else ''
    })

results_df = pd.DataFrame(results_rows)
results_df = results_df.sort_values(by='均值差(风化-未风化, CLR)', key=lambda s: s.abs(), ascending=False)

print("\n风化效应量化（CLR 空间）：")
print(results_df.to_string(index=False))
    

# =============================================================
# 2.3 风化逆转预测模型 (PLS Regression Model)
# =============================================================
print("\n2.3 建立风化逆转预测模型 (PLS):")

# 准备训练数据
X_train_pls = weathered_paired[clr_cols]
y_train_pls = unweathered_paired[clr_cols]

# 清洗训练集：去除任何包含缺失值的配对样本，确保 PLS 能正常训练
train_joined = pd.concat([X_train_pls, y_train_pls.add_suffix("__target")], axis=1).dropna()
X_train_pls = train_joined[clr_cols]
y_train_pls = train_joined[[c + "__target" for c in clr_cols]]
y_train_pls.columns = clr_cols  # 还原列名以匹配后续使用

# 自适应设定 PLS 组分数：受限于样本量、特征数与输出维度
n_components_upper = int(min(
    max(1, X_train_pls.shape[0] - 1),  # n_samples - 1 至少为 1
    X_train_pls.shape[1],              # n_features
    y_train_pls.shape[1]               # n_targets
))
if n_components_upper < 1:
    raise ValueError("配对样本数量不足，无法训练 PLS 模型。请检查配对样本是否齐全。")

n_components = min(10, n_components_upper)
print(f"PLS 使用的组分数: {n_components} (上限 {n_components_upper})")

# 建立并训练 PLS 模型
pls = PLSRegression(n_components=n_components)
pls.fit(X_train_pls, y_train_pls)

# 评估模型性能
y_pred_pls = pls.predict(X_train_pls)
r2 = r2_score(y_train_pls, y_pred_pls)
print(f"PLS模型在训练集上的 R² score: {r2:.4f}")

# --- 应用模型进行预测 ---
# 选取一个风化样本作为例子，如 "02"
sample_02_weathered = final_classified_df[final_classified_df['文物采样点'] == '02']

if not sample_02_weathered.empty:
    # 提取其CLR变换后的数据
    sample_02_clr = sample_02_weathered[clr_cols]
    
    # 使用训练好的PLS模型预测其风化前的CLR值
    predicted_clr = pls.predict(sample_02_clr)
    
    # 逆变换回原始百分比成分
    predicted_composition = inverse_clr_transform(predicted_clr)
    
    print("\n示例：预测样本'02'风化前的化学成分:")
    original_comp_df = pd.DataFrame(sample_02_weathered[chemical_cols].values, columns=chemical_cols, index=['测量值 (风化)'])
    predicted_comp_df = pd.DataFrame(predicted_composition, columns=chemical_cols, index=['预测值 (风化前)'])
    
    # 显示关键成分的变化
    display_cols = ['二氧化硅(SiO2)', '氧化钾(K2O)', '氧化铅(PbO)', '氧化钡(BaO)']
    print(pd.concat([original_comp_df[display_cols], predicted_comp_df[display_cols]]))