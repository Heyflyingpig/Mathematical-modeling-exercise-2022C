import pandas as pd
import numpy as np
import re
import os

# ==================================
# 1. 数据加载与合并 (Data Loading and Merging)
# ==================================
# 加载三个表单
basepath = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(basepath, 'C题')

def read_csv_robust(filename, encodings=(
    'gb18030',  # 覆盖范围广，兼容 GBK/GB2312
    'gbk',      # 常见中文编码
    'utf-8-sig',# 处理含 BOM 的 UTF-8
    'utf-8'     # 纯 UTF-8
)):
    """
    从 `C题` 目录稳健读取 CSV 文件：依次尝试多种编码，读取成功即返回。
    - filename: 文件名，例如 '表单1.csv'
    - encodings: 依次尝试的编码列表
    """
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到数据文件: {file_path}")

    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise RuntimeError(
        f"无法用这些编码读取文件 {file_path}: {list(encodings)}; 最后错误: {last_error}"
    )

# 读取三个表单（固定目录 + 多编码尝试）
df1 = read_csv_robust('表单1.csv')
df2 = read_csv_robust('表单2.csv')
df3_unknown = read_csv_robust('表单3.csv')


# --- 数据清洗与合并 ---
# 函数：从“文物采样点”中提取“文物编号”
def get_wenwu_id(s):
    match = re.match(r'(\d+)', str(s))
    return int(match.group(1)) if match else None

# 为表单2创建“文物编号”列，用于合并
df2['文物编号'] = df2['文物采样点'].apply(get_wenwu_id)

# 将表单1和表单2合并。表单2中的每个采样点都是一个独立样本。
# 我们使用左合并，以df2为基础，补充df1中的信息。
merged_df = pd.merge(df2, df1, on='文物编号', how='left')

# 提取化学成分列的列名
chemical_cols = df2.columns.drop(['文物采样点', '文物编号'])

# ==================================
# 2. 数据预处理函数 (Data Preprocessing Functions)
# ==================================
# --- 零值处理：乘法替换法 (Multiplicative Replacement for Zeros) ---
# 该方法用一个小值δ替换零值，并按比例缩放其他值，以保持数据结构。
def multiplicative_replacement(df, cols, delta=1e-5):
    """
    对DataFrame中的指定列应用乘法替换。
    df: a pandas DataFrame.
    cols: a list of column names for compositional data.
    delta: a small value to replace zeros.
    """
    df_copy = df.copy()
    # 仅对成分数据进行操作
    comp_data = df_copy[cols].values
    
    # 对每一行（样本）进行处理
    for i in range(comp_data.shape[0]):
        row = comp_data[i]
        zeros = (row == 0) | np.isnan(row)
        non_zeros = ~zeros
        
        c = np.sum(zeros) # 零值的数量
        if c > 0:
            # 用delta替换零值
            row[zeros] = delta
            
            # 按比例缩放非零值
            s_nonzero = np.sum(row[non_zeros])
            row[non_zeros] = row[non_zeros] * (1 - c * delta) / s_nonzero
            
        # 将处理后的行写回
        comp_data[i] = row
        
    df_copy[cols] = comp_data
    return df_copy

# --- CLR变换 (Centered Log-Ratio Transformation) ---
# 这是成分数据分析的核心，将数据从约束空间转换到欧几里得空间。
def clr_transform(df, cols):
    """
    对DataFrame中的指定列进行CLR变换。
    """
    df_processed = multiplicative_replacement(df, cols)
    comp_data = df_processed[cols].values
    
    # 避免log(0)
    # 乘法替换已经处理了零值，但以防万一
    comp_data[comp_data <= 0] = 1e-9 
    
    # 计算几何平均值
    geom_mean = np.exp(np.mean(np.log(comp_data), axis=1, keepdims=True))
    
    # CLR变换
    clr_data = np.log(comp_data / geom_mean)
    
    clr_df = pd.DataFrame(clr_data, columns=[f"CLR_{col}" for col in cols], index=df.index)
    return clr_df

# --- 逆CLR变换 (Inverse CLR Transformation) ---
# 将CLR变换后的数据还原为原始比例。
def inverse_clr_transform(clr_data):
    """
    对CLR变换后的数据进行逆变换。
    clr_data: a numpy array or DataFrame of CLR-transformed data.
    """
    if isinstance(clr_data, pd.DataFrame):
        clr_data = clr_data.values
        
    exp_data = np.exp(clr_data)
    sum_exp = np.sum(exp_data, axis=1, keepdims=True)
    
    # 归一化并乘以100，得到百分比
    original_composition = (exp_data / sum_exp) * 100
    return original_composition


# --- 应用预处理 ---
# 对已分类数据应用CLR变换
# 填充NaN值为0，因为它们代表“未检测到”
merged_df[chemical_cols] = merged_df[chemical_cols].fillna(0)
clr_df_classified = clr_transform(merged_df, chemical_cols)
# 合并CLR变换后的数据和原始标签信息
final_classified_df = pd.concat([merged_df.reset_index(drop=True), clr_df_classified.reset_index(drop=True)], axis=1)

print("数据准备完成。已分类的数据集包含 {} 个样本。".format(len(final_classified_df)))
print("已分类数据预览:")
print(final_classified_df[['文物采样点', '类型', '表面风化', 'CLR_二氧化硅(SiO2)']].head())
