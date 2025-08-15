import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import preprocess
import Q1

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入已预处理的数据
final_classified_df = preprocess.final_classified_df
chemical_cols = preprocess.chemical_cols
clr_transform = preprocess.clr_transform
multiplicative_replacement = preprocess.multiplicative_replacement
inverse_clr_transform = preprocess.inverse_clr_transform
df3_unknown = preprocess.df3_unknown

# 从Q1导入PLS模型
pls_model = Q1.pls

print("\n--- 问题3：未知类别玻璃文物鉴定与敏感性分析 ---\n")

# =============================================================
# 3.1 构建玻璃类型分类模型
# =============================================================
print("3.1 构建玻璃类型分类模型:")

# 准备训练数据：使用无风化样本避免风化干扰
training_data = final_classified_df[final_classified_df['表面风化'] == '无风化'].copy()
print(f"训练样本数量（无风化）: {len(training_data)}")

# 特征和标签
clr_cols = [f"CLR_{col}" for col in chemical_cols]
X_train = training_data[clr_cols].values
y_train = training_data['类型'].values

# 建立随机森林分类器
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# 训练模型
rf_classifier.fit(X_train, y_train)

# 交叉验证评估
cv_scores = cross_val_score(
    rf_classifier, X_train, y_train, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)

print(f"交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")

# 特征重要性分析
feature_importance = pd.DataFrame({
    '化学成分': chemical_cols,
    '重要性得分': rf_classifier.feature_importances_
}).sort_values('重要性得分', ascending=False)

print("\n特征重要性排序（前8位）:")
print(feature_importance.head(8).to_string(index=False))

# =============================================================
# 3.2 预处理表单3未知样本
# =============================================================
print("\n3.2 预处理表单3未知样本:")

# 处理表单3数据
df3_processed = df3_unknown.copy()
# 填充缺失值为0（表示未检测到）
df3_processed[chemical_cols] = df3_processed[chemical_cols].fillna(0)

print("表单3样本信息:")
for idx, row in df3_processed.iterrows():
    print(f"{row['文物编号']}: {row['表面风化']}")

# =============================================================
# 3.3 处理风化样本
# =============================================================
print("\n3.3 处理风化样本（预测风化前成分）:")

# 识别风化样本
weathered_samples = df3_processed[df3_processed['表面风化'] == '风化'].copy()
unweathered_samples = df3_processed[df3_processed['表面风化'] == '无风化'].copy()

print(f"风化样本数量: {len(weathered_samples)}")
print(f"无风化样本数量: {len(unweathered_samples)}")

# 对风化样本应用CLR变换并使用PLS模型预测风化前成分
processed_samples_list = []

if len(weathered_samples) > 0:
    # 对风化样本进行CLR变换
    weathered_clr = clr_transform(weathered_samples, chemical_cols)
    
    # 使用PLS模型预测风化前的CLR值
    predicted_clr = pls_model.predict(weathered_clr[clr_cols])
    
    # 逆变换回原始百分比成分
    predicted_compositions = inverse_clr_transform(predicted_clr)
    
    # 创建预测后的DataFrame
    for i, (idx, row) in enumerate(weathered_samples.iterrows()):
        predicted_row = row.copy()
        predicted_row[chemical_cols] = predicted_compositions[i]
        processed_samples_list.append(predicted_row)
        
        print(f"\n样本 {row['文物编号']} 风化前成分预测:")
        comparison_df = pd.DataFrame({
            '测量值(风化)': weathered_samples.loc[idx, chemical_cols],
            '预测值(风化前)': predicted_compositions[i]
        }, index=chemical_cols)
        
        # 显示主要成分
        main_components = ['二氧化硅(SiO2)', '氧化钾(K2O)', '氧化铅(PbO)', '氧化钡(BaO)']
        print(comparison_df.loc[main_components].round(3))

# 对无风化样本直接使用测量值
for idx, row in unweathered_samples.iterrows():
    processed_samples_list.append(row)

# 合并所有处理后的样本
df3_final = pd.DataFrame(processed_samples_list).reset_index(drop=True)

# =============================================================
# 3.4 对未知样本进行类型预测
# =============================================================
print("\n3.4 未知样本类型预测:")

# 对处理后的数据进行CLR变换
df3_clr = clr_transform(df3_final, chemical_cols)
X_unknown = df3_clr[clr_cols].values

# 进行预测
predictions = rf_classifier.predict(X_unknown)
prediction_probs = rf_classifier.predict_proba(X_unknown)

# 获取类别标签
class_labels = rf_classifier.classes_

# 整理预测结果
results = []
for i, (idx, row) in enumerate(df3_final.iterrows()):
    prob_dict = dict(zip(class_labels, prediction_probs[i]))
    max_prob = max(prediction_probs[i])
    
    results.append({
        '文物编号': row['文物编号'],
        '表面风化': row['表面风化'],
        '预测类型': predictions[i],
        '预测概率': max_prob,
        '高钾概率': prob_dict.get('高钾', 0),
        '铅钡概率': prob_dict.get('铅钡', 0)
    })

results_df = pd.DataFrame(results)
print("\n预测结果:")
print(results_df.to_string(index=False))

# =============================================================
# 3.5 敏感性分析
# =============================================================
print("\n3.5 敏感性分析:")

def sensitivity_analysis(sample_clr, classifier, n_iterations=100, noise_level=0.05):
    """
    对单个样本进行敏感性分析
    """
    original_pred = classifier.predict([sample_clr])[0]
    original_prob = classifier.predict_proba([sample_clr])[0].max()
    
    # 计算CLR数据的标准差用于生成噪声
    noise_std = np.std(sample_clr) * noise_level
    
    stable_count = 0
    prob_variations = []
    
    for _ in range(n_iterations):
        # 添加随机噪声
        noise = np.random.normal(0, noise_std, size=sample_clr.shape)
        perturbed_sample = sample_clr + noise
        
        # 进行预测
        pred = classifier.predict([perturbed_sample])[0]
        prob = classifier.predict_proba([perturbed_sample])[0].max()
        
        prob_variations.append(prob)
        
        if pred == original_pred:
            stable_count += 1
    
    stability_score = (stable_count / n_iterations) * 100
    prob_std = np.std(prob_variations)
    
    return stability_score, prob_std

# 对每个未知样本进行敏感性分析
sensitivity_results = []

print("\n逐个样本敏感性分析:")
for i, (idx, row) in enumerate(df3_final.iterrows()):
    sample_clr = X_unknown[i]
    stability, prob_std = sensitivity_analysis(sample_clr, rf_classifier)
    
    sensitivity_results.append({
        '文物编号': row['文物编号'],
        '表面风化': row['表面风化'],
        '预测类型': predictions[i],
        '预测概率': prediction_probs[i].max(),
        '分类稳定性(%)': stability,
        '概率标准差': prob_std
    })
    
    print(f"样本 {row['文物编号']}: 稳定性 {stability:.0f}%, 概率标准差 {prob_std:.3f}")

sensitivity_df = pd.DataFrame(sensitivity_results)

# =============================================================
# 3.6 综合结果展示
# =============================================================
print("\n3.6 综合分类结果与敏感性分析:")

final_results = pd.merge(results_df, sensitivity_df[['文物编号', '分类稳定性(%)', '概率标准差']], on='文物编号')

# 添加置信度评级
def confidence_rating(prob, stability):
    if prob >= 0.95 and stability >= 95:
        return "极高"
    elif prob >= 0.85 and stability >= 85:
        return "高"
    elif prob >= 0.70 and stability >= 70:
        return "中等"
    else:
        return "低"

final_results['置信度'] = final_results.apply(
    lambda x: confidence_rating(x['预测概率'], x['分类稳定性(%)']), axis=1
)

print("\n最终分类结果：")
display_cols = ['文物编号', '表面风化', '预测类型', '预测概率', '分类稳定性(%)', '置信度']
print(final_results[display_cols].round(3).to_string(index=False))

# =============================================================
# 3.7 可视化分析
# =============================================================
print("\n3.7 生成可视化图表...")

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 预测概率分布
axes[0, 0].bar(final_results['文物编号'], final_results['预测概率'], 
               color=['red' if x == '铅钡' else 'blue' for x in final_results['预测类型']])
axes[0, 0].set_title('各样本预测概率')
axes[0, 0].set_xlabel('文物编号')
axes[0, 0].set_ylabel('预测概率')
axes[0, 0].axhline(y=0.8, color='orange', linestyle='--', label='高置信度阈值')
axes[0, 0].legend()

# 2. 稳定性分析
axes[0, 1].bar(final_results['文物编号'], final_results['分类稳定性(%)'],
               color=['red' if x == '铅钡' else 'blue' for x in final_results['预测类型']])
axes[0, 1].set_title('分类稳定性分析')
axes[0, 1].set_xlabel('文物编号')
axes[0, 1].set_ylabel('稳定性 (%)')
axes[0, 1].axhline(y=85, color='orange', linestyle='--', label='高稳定性阈值')
axes[0, 1].legend()

# 3. 特征重要性
top_features = feature_importance.head(8)
axes[1, 0].barh(range(len(top_features)), top_features['重要性得分'])
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['化学成分'])
axes[1, 0].set_title('关键化学成分重要性')
axes[1, 0].set_xlabel('重要性得分')

# 4. 置信度分布
confidence_counts = final_results['置信度'].value_counts()
axes[1, 1].pie(confidence_counts.values, labels=confidence_counts.index, autopct='%1.1f%%')
axes[1, 1].set_title('预测置信度分布')

plt.tight_layout()
plt.savefig('Q3_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================
# 3.8 详细敏感性分析报告
# =============================================================
print("\n3.8 详细敏感性分析报告:")

print("\n各样本的分类可靠性评估:")
for _, row in final_results.iterrows():
    status = ""
    if row['置信度'] == "极高":
        status = "分类结果极为可靠，预测概率高且对扰动不敏感"
    elif row['置信度'] == "高":
        status = "分类结果可靠，具有较高的置信度"
    elif row['置信度'] == "中等":
        status = "分类结果基本可靠，但存在一定不确定性"
    else:
        status = "分类结果不确定，建议进一步验证"
    
    print(f"样本 {row['文物编号']}: {row['预测类型']} (概率:{row['预测概率']:.3f}, 稳定性:{row['分类稳定性(%)']:.1f}%) - {status}")

print(f"\n整体分析总结:")
print(f"- 共分析未知样本: {len(final_results)} 个")
print(f"- 预测为高钾玻璃: {sum(final_results['预测类型'] == '高钾')} 个")
print(f"- 预测为铅钡玻璃: {sum(final_results['预测类型'] == '铅钡')} 个")
print(f"- 高置信度预测: {sum(final_results['置信度'].isin(['极高', '高']))} 个")
print(f"- 平均预测概率: {final_results['预测概率'].mean():.3f}")
print(f"- 平均稳定性: {final_results['分类稳定性(%)'].mean():.1f}%")
