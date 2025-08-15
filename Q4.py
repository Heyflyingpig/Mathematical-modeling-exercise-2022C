#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q4_corr_diff_only.py — 2022C 题 问题四：不同类别的成分关联差异（严格按“思路文档”）
----------------------------------------------------------------------------------
目标：
1) 在 CLR 空间分别计算“高钾玻璃 / 铅钡玻璃”的 Pearson 相关矩阵；
2) 计算差异矩阵 D = Corr(铅钡) − Corr(高钾)，并绘制热力图；
3) 输出 |D| 最大的成分对（Top-k），以及 Fisher z 检验的显著性（两独立相关系数之差）。
（可选）--boot > 0 时，对各类做 bootstrap 重采样以估计 D 的均值/标准差，评估稳健性。

依赖：pandas, numpy, matplotlib（仅此；不使用 seaborn）
使用：
    python Q4_corr_diff_only.py --base_dir C题 --topk 12 --boot 0
"""
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
import matplotlib.pyplot as plt
import preprocess

# --------------------------- 成分工具 ---------------------------

def detect_comp_cols(df: pd.DataFrame):
    keys = ["SiO", "Al2O", "K2O", "Na2O", "CaO", "MgO",
            "PbO", "BaO", "Fe2O", "SrO", "TiO", "MnO",
            "CuO", "ZnO", "SnO", "Sb2O", "P2O", "SO", "Cl"]
    out = []
    for c in df.columns:
        s = str(c)
        if any(k in s for k in keys) or (("O" in s) and any(ch.isdigit() for ch in s)):
            out.append(c)
    seen=set(); cols=[]
    for c in out:
        if c not in seen:
            cols.append(c); seen.add(c)
    return cols

def closure(X: np.ndarray, total: float = 100.0) -> np.ndarray:
    s = X.sum(axis=1, keepdims=True)
    s[s==0] = 1.0
    return X / s * total

def multiplicative_replacement(X: np.ndarray, delta: float = 1e-5) -> np.ndarray:
    X = X.copy()
    zeros = (X <= 0)
    if np.any(zeros):
        X[zeros] = delta
    return closure(X, 100.0)

def clr(X: np.ndarray) -> np.ndarray:
    X = multiplicative_replacement(closure(X, 100.0), delta=1e-5)
    g = np.exp(np.log(X).mean(axis=1, keepdims=True))
    return np.log(X / g)

def valid_mask(df: pd.DataFrame, comp_cols):
    s = df[comp_cols].fillna(0).sum(axis=1)
    return (s >= 85.0) & (s <= 105.0)

# --------------------------- IO ---------------------------

def read_forms(base_dir: str):
    base = Path(base_dir)
    f1 = base / "表单1.csv"
    f2 = base / "表单2.csv"
    df1 = pd.read_csv(f1, encoding="utf-8")
    try:
        df2 = pd.read_csv(f2, encoding="utf-8")
    except UnicodeDecodeError:
        df2 = pd.read_csv(f2, encoding="gbk")
    return df1, df2

def find_cols(df1: pd.DataFrame):
    id_col = next((c for c in df1.columns if "编号" in str(c)), None)
    type_col = next((c for c in df1.columns if ("类型" in str(c) or "类别" in str(c))), None)
    if id_col is None or type_col is None:
        raise ValueError("表单1中未找到‘编号’或‘类型/类别’列")
    return id_col, type_col

def match_id_col(df: pd.DataFrame, id_col1: str):
    return id_col1 if id_col1 in df.columns else next((c for c in df.columns if "编号" in str(c)), None)

# --------------------------- 统计与绘图 ---------------------------

def fisher_z(r):
    # clip to valid range to avoid inf
    r = max(min(r, 0.999999), -0.999999)
    return 0.5 * math.log((1+r)/(1-r))

def norm_cdf(x):
    # standard normal CDF via erf
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def correlation_by_class(df1, df2, comp_cols):
    id_col, type_col = find_cols(df1)
    id2 = match_id_col(df2, id_col)
    meta = df1[[id_col, type_col]].copy().rename(columns={id_col:"ID", type_col:"TYPE"})
    data = df2.copy()
    if id2 != "ID":
        data = data.rename(columns={id2:"ID"})
    df = data.merge(meta, on="ID", how="left")
    df = df.loc[valid_mask(df, comp_cols)].reset_index(drop=True)

    def map_cls(s):
        s = str(s)
        if ("铅" in s) and ("钡" in s):
            return "铅钡玻璃"
        if ("高钾" in s) or ("钾" in s):
            return "高钾玻璃"
        return np.nan

    df["CLS"] = df["TYPE"].map(map_cls)
    df = df.dropna(subset=["CLS"]).copy()

    X = df[comp_cols].fillna(0).to_numpy(float)
    Z = clr(closure(X, 100.0))
    dfZ = pd.DataFrame(Z, columns=comp_cols)
    dfZ["CLS"] = df["CLS"].values

    Z_hk = dfZ[dfZ["CLS"]=="高钾玻璃"][comp_cols].to_numpy()
    Z_pb = dfZ[dfZ["CLS"]=="铅钡玻璃"][comp_cols].to_numpy()

    if Z_hk.shape[0] < 5 or Z_pb.shape[0] < 5:
        raise ValueError("两类样本数不足（<5）无法稳定估计相关矩阵。")

    C_hk = np.corrcoef(Z_hk, rowvar=False)
    C_pb = np.corrcoef(Z_pb, rowvar=False)
    return C_hk, C_pb, Z_hk.shape[0], Z_pb.shape[0]

def plot_heatmap(mat, labels, title, outpath):
    fig = plt.figure(figsize=(8,6))
    plt.imshow(mat, vmin=-1, vmax=1)
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=60)
    plt.yticks(range(len(labels)), labels)
    # 仅在上三角打印数值，避免遮挡
    n = len(labels)
    for i in range(n):
        for j in range(i+1, n):
            plt.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6)
    plt.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def top_differences(C_hk, C_pb, comp_cols, n1, n2, topk=12):
    D = C_pb - C_hk
    pairs = []
    pvals = []
    n = len(comp_cols)
    for i in range(n):
        for j in range(i+1, n):
            r1 = float(C_hk[i,j]); r2 = float(C_pb[i,j])
            d = float(r2 - r1)
            # Fisher z test for difference between two independent correlations
            # z = (z2 - z1)/sqrt(1/(n2-3)+1/(n1-3))
            if n1 > 3 and n2 > 3:
                z1 = fisher_z(r1); z2 = fisher_z(r2)
                se = math.sqrt(1.0/(n1-3) + 1.0/(n2-3))
                z = (z2 - z1)/se
                p = 2.0 * (1.0 - norm_cdf(abs(z)))
            else:
                p = float('nan')
            pairs.append((comp_cols[i], comp_cols[j], r1, r2, d, abs(d), p))
    # 排序：先按 |D| 降序，再按 p 值升序（显著性更高优先）
    pairs.sort(key=lambda x: (-x[5], x[6] if not math.isnan(x[6]) else 1.0))
    df = pd.DataFrame(pairs, columns=["成分1","成分2","Corr_高钾","Corr_铅钡","D=铅钡-高钾","|D|","p值(Fisher z)"])
    top_df = df.head(topk).copy()
    return D, df, top_df

def bootstrap_D(df1, df2, comp_cols, B=200, seed=2024):
    # 返回 D 的均值与标准差矩阵（可选）
    rng = np.random.RandomState(seed)
    id_col, type_col = find_cols(df1)
    id2 = match_id_col(df2, id_col)
    meta = df1[[id_col, type_col]].copy().rename(columns={id_col:"ID", type_col:"TYPE"})
    data = df2.copy()
    if id2 != "ID":
        data = data.rename(columns={id2:"ID"})
    df = data.merge(meta, on="ID", how="left")
    df = df.loc[valid_mask(df, comp_cols)].reset_index(drop=True)

    def map_cls(s):
        s = str(s)
        if ("铅" in s) and ("钡" in s): return "铅钡玻璃"
        if ("高钾" in s) or ("钾" in s): return "高钾玻璃"
        return np.nan

    df["CLS"] = df["TYPE"].map(map_cls)
    df = df.dropna(subset=["CLS"]).copy()
    X = df[comp_cols].fillna(0).to_numpy(float)
    Z = clr(closure(X, 100.0))
    dfZ = pd.DataFrame(Z, columns=comp_cols); dfZ["CLS"]=df["CLS"].values

    Z_hk = dfZ[dfZ["CLS"]=="高钾玻璃"][comp_cols].to_numpy()
    Z_pb = dfZ[dfZ["CLS"]=="铅钡玻璃"][comp_cols].to_numpy()
    n1, n2 = Z_hk.shape[0], Z_pb.shape[0]

    if n1 < 5 or n2 < 5:
        raise ValueError("两类样本数不足，无法做 bootstrap 估计。")

    mats = []
    for b in range(B):
        idx1 = rng.randint(0, n1, size=n1)
        idx2 = rng.randint(0, n2, size=n2)
        C1 = np.corrcoef(Z_hk[idx1], rowvar=False)
        C2 = np.corrcoef(Z_pb[idx2], rowvar=False)
        mats.append(C2 - C1)
    mats = np.stack(mats, axis=0)  # (B, p, p)
    D_mean = np.nanmean(mats, axis=0)
    D_std = np.nanstd(mats, axis=0, ddof=1)
    return D_mean, D_std

# --------------------------- 主流程 ---------------------------

def main(base_dir: str, topk: int, boot: int, seed: int):
    outdir = Path("outputs_q4"); outdir.mkdir(parents=True, exist_ok=True)
    df1 = preprocess.df1
    df2 = preprocess.df2
    comp_cols = detect_comp_cols(df2)
    if not comp_cols:
        raise ValueError("未识别到成分列")
    C_hk, C_pb, n1, n2 = correlation_by_class(df1, df2, comp_cols)

    # 保存矩阵
    pd.DataFrame(C_hk, index=comp_cols, columns=comp_cols).to_csv(outdir / "Q4_corr_高钾.csv", encoding="utf-8-sig")
    pd.DataFrame(C_pb, index=comp_cols, columns=comp_cols).to_csv(outdir / "Q4_corr_铅钡.csv", encoding="utf-8-sig")

    # 差异与Top对
    D, df_allpairs, top_df = top_differences(C_hk, C_pb, comp_cols, n1, n2, topk=topk)
    pd.DataFrame(D, index=comp_cols, columns=comp_cols).to_csv(outdir / "Q4_corr_差异矩阵铅钡减高钾.csv", encoding="utf-8-sig")
    df_allpairs.to_csv(outdir / "Q4_所有成分对_差异与p值.csv", index=False, encoding="utf-8-sig")
    top_df.to_csv(outdir / f"Q4_差异Top{topk}对.csv", index=False, encoding="utf-8-sig")

    # 绘图（Matplotlib，单图，无设定颜色）
    plot_heatmap(C_hk, comp_cols, "高钾玻璃：CLR相关矩阵", outdir / "Q4_heatmap_高钾.png")
    plot_heatmap(C_pb, comp_cols, "铅钡玻璃：CLR相关矩阵", outdir / "Q4_heatmap_铅钡.png")
    plot_heatmap(D, comp_cols, "差异矩阵 D = Corr(铅钡) - Corr(高钾)", outdir / "Q4_heatmap_差异矩阵.png")

    # （可选）bootstrap 稳健性
    if boot and boot > 0:
        D_mean, D_std = bootstrap_D(df1, df2, comp_cols, B=boot, seed=seed)
        pd.DataFrame(D_mean, index=comp_cols, columns=comp_cols).to_csv(outdir / "Q4_bootstrap_D均值.csv", encoding="utf-8-sig")
        pd.DataFrame(D_std, index=comp_cols, columns=comp_cols).to_csv(outdir / "Q4_bootstrap_D标准差.csv", encoding="utf-8-sig")

    # 摘要
    with open(outdir / "Q4_结果摘要.md", "w", encoding="utf-8") as f:
        f.write("# 问题四：不同类别的成分关联差异（按思路文档实现）\n\n")
        f.write(f"- 样本量：高钾 n={n1}，铅钡 n={n2}\n")
        f.write("- 相关矩阵：见 `Q4_corr_高钾.csv`、`Q4_corr_铅钡.csv`；差异矩阵：`Q4_corr_差异矩阵铅钡减高钾.csv`。\n")
        f.write(f"- |D| 最大的前 {topk} 对见 `Q4_差异Top{topk}对.csv`；完整成分对与显著性见 `Q4_所有成分对_差异与p值.csv`。\n")
        if boot and boot > 0:
            f.write(f"- Bootstrap(B={boot}) 稳健性：`Q4_bootstrap_D均值.csv`、`Q4_bootstrap_D标准差.csv`。\n")

    print("完成。输出目录：", str(outdir.resolve()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="C题")
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--boot", type=int, default=0, help="bootstrap 次数（默认0=不启用）")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    main(args.base_dir, args.topk, args.boot, args.seed)
