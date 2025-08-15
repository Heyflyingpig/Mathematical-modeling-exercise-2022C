#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2_hierRF.py — 2022C 题（古代玻璃制品）问题二：按“思路文档”实施的分类规律与亚类划分 + 敏感性分析
---------------------------------------------------------------------------------
实现要点（严格对齐你的思路文档）：
1) 成分数据 CoDA：85%~105% 有效性筛选；零值“乘法替代”；闭合到100%；CLR 变换（主要工作空间）。
2) 监督分类：仅用“无风化”样本训练 RandomForest，分层K折CV，报告 Accuracy/Precision/Recall/F1；输出特征重要性。
3) 亚类划分：分别在（高钾/铅钡）子集上，按推荐变量集做 Ward 凝聚层次聚类；用轮廓系数在 k∈[2,5] 内选最佳；给出几何均值“化学指纹”。
4) 敏感性：
   a) 联动法敏感性：Ward vs Average，报告 ARI；
   b) 自助法敏感性：bootstrap 多次聚类，输出共聚频率矩阵与稳定性指标（例如每个簇的平均共聚概率）。
5) 全部结果写入 outputs_q2/ 目录并生成“可直接入文”的 Markdown 摘要。

使用：
$ python Q2_hierRF.py --base_dir C题 --seed 2024 --cv 5 --boot 100

依赖：pandas, numpy, scikit-learn, scipy, matplotlib
"""
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             accuracy_score, f1_score, precision_score, recall_score,
                             silhouette_score)
from sklearn.ensemble import RandomForestClassifier

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score

import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
import matplotlib.pyplot as plt
import preprocess

# --------------------------- 成分工具 ---------------------------

def detect_comp_cols(df: pd.DataFrame) -> List[str]:
    keys = ["SiO", "Al2O", "K2O", "Na2O", "CaO", "MgO", "PbO", "BaO", "Fe2O", "SrO", "TiO", "MnO", "CuO", "ZnO", "SnO", "Sb2O", "P2O", "SO", "Cl"]
    out = []
    for c in df.columns:
        s = str(c)
        if any(k in s for k in keys):
            out.append(c)
        elif ("O" in s) and any(ch.isdigit() for ch in s):
            out.append(c)
    # 去重保持顺序
    seen = set(); cols = []
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
        c = zeros.sum(axis=1, keepdims=True).astype(float)
        # 零 -> delta，非零整体缩放，保持形状（简化版本）
        X[zeros] = delta
        # 重新闭合
    return closure(X, 100.0)

def clr(X: np.ndarray) -> np.ndarray:
    X = multiplicative_replacement(closure(X, 100.0), delta=1e-5)
    g = np.exp(np.log(X).mean(axis=1, keepdims=True))
    return np.log(X / g)

def inv_clr(Z: np.ndarray, total: float = 100.0) -> np.ndarray:
    X = np.exp(Z)
    return closure(X, total)

# --------------------------- IO ---------------------------

def valid_mask(df2: pd.DataFrame, comp_cols: List[str]) -> np.ndarray:
    s = df2[comp_cols].fillna(0).sum(axis=1)
    return (s >= 85.0) & (s <= 105.0)

# --------------------------- 分类（RandomForest, CV） ---------------------------

def train_rf_cv(X: np.ndarray, y: np.ndarray, cv: int = 5, seed: int = 2024):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    clf = RandomForestClassifier(n_estimators=600, max_depth=None, random_state=seed, n_jobs=-1, class_weight="balanced_subsample")
    y_pred = cross_val_predict(clf, X, y, cv=skf, method="predict")
    y_prob = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:,1]
    clf.fit(X, y)
    metrics = dict(
        acc = float(accuracy_score(y, y_pred)),
        prec = float(precision_score(y, y_pred)),
        rec = float(recall_score(y, y_pred)),
        f1 = float(f1_score(y, y_pred)),
        auc = float(roc_auc_score(y, y_prob))
    )
    importances = clf.feature_importances_
    return metrics, importances

# --------------------------- 层次聚类与评估 ---------------------------

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial.distance import pdist

def hierarchical_cluster(Z: np.ndarray, k: int, method: str = "ward"):
    D = pdist(Z, metric="euclidean")
    L = linkage(D, method=method)
    labels = fcluster(L, t=k, criterion="maxclust")
    return labels, L

def choose_k_by_silhouette(Z: np.ndarray, kmin=2, kmax=5, method="ward"):
    best_k, best_score, best_labels, best_link = None, -1.0, None, None
    for k in range(kmin, kmax+1):
        labels, L = hierarchical_cluster(Z, k=k, method=method)
        try:
            sc = silhouette_score(Z, labels, metric="euclidean")
        except Exception:
            sc = -1.0
        if sc > best_score:
            best_k, best_score, best_labels, best_link = k, sc, labels, L
    return best_k, best_score, best_labels, best_link

def geo_mean_fingerprint(X_closed: np.ndarray, labels: np.ndarray, comp_cols: List[str]) -> pd.DataFrame:
    Z = clr(X_closed)
    rows = []
    for k in sorted(np.unique(labels)):
        idx = np.where(labels == k)[0]
        Zk = Z[idx]
        center = Zk.mean(axis=0, keepdims=True)
        comp = inv_clr(center, total=100.0).ravel()
        row = dict(subcluster=int(k), size=int(idx.size))
        row.update({c: float(v) for c, v in zip(comp_cols, comp)})
        rows.append(row)
    return pd.DataFrame(rows)

def cophenetic_plot(L, outpath: Path, title: str):
    fig = plt.figure(figsize=(8,4))
    dendrogram(L, no_labels=True)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def bootstrap_coclustering(Z: np.ndarray, k: int, B: int = 100, method: str = "ward", seed: int = 2024) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n = Z.shape[0]
    co = np.zeros((n, n), dtype=float)
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        uniq, inv = np.unique(idx, return_inverse=True)
        Zb = Z[uniq]
        labels_b, _ = hierarchical_cluster(Zb, k=k, method=method)
        for a in range(len(uniq)):
            for bb in range(a+1, len(uniq)):
                if labels_b[a] == labels_b[bb]:
                    ia = uniq[a]; ib = uniq[bb]
                    co[ia, ib] += 1.0
                    co[ib, ia] += 1.0
    co /= float(B)
    np.fill_diagonal(co, 1.0)
    return co

# --------------------------- 主流程 ---------------------------

def main(base_dir: str, seed: int, cv: int, boot: int):
    outdir = Path("outputs_q2")
    outdir.mkdir(parents=True, exist_ok=True)

    df1 = preprocess.df1
    df2 = preprocess.df2
    id_col = next((c for c in df1.columns if "编号" in str(c)), None)
    type_col = next((c for c in df1.columns if ("类型" in str(c) or "类别" in str(c))), None)
    weather_col = next((c for c in df1.columns if "风化" in str(c)), None)
    if id_col is None or type_col is None:
        raise ValueError("表单1中未找到‘编号’或‘类型/类别’列")
    if weather_col is None:
        weather_col = ""

    id_col2 = id_col if id_col in df2.columns else next((c for c in df2.columns if "编号" in str(c)), None)
    if id_col2 is None:
        raise ValueError("表单2无法匹配编号列")

    meta = df1[[id_col, type_col] + ([weather_col] if weather_col else [])].copy()
    meta = meta.rename(columns={id_col: "ID", type_col: "TYPE"})
    if weather_col:
        meta = meta.rename(columns={weather_col: "WEATHER"})
    data = df2.copy()
    if id_col2 != "ID":
        data = data.rename(columns={id_col2: "ID"})
    df = data.merge(meta, on="ID", how="left")

    comp_cols = detect_comp_cols(df)
    if not comp_cols:
        raise ValueError("未识别到成分列")
    mask_valid = valid_mask(df, comp_cols)
    df = df.loc[mask_valid].reset_index(drop=True)

    if "WEATHER" in df.columns:
        non_weather_vals = [v for v in ["无风化", "否", "未风化", "无", "0", 0] if v in set(df["WEATHER"].astype(str))]
        if non_weather_vals:
            df_tr = df[df["WEATHER"].astype(str).isin(non_weather_vals)].copy()
        else:
            df_tr = df.copy()
    else:
        df_tr = df.copy()

    def map_label(s):
        s = str(s)
        if "铅" in s and "钡" in s:
            return 1
        if "高钾" in s or "钾" in s:
            return 0
        return np.nan
    df_tr["Y"] = df_tr["TYPE"].map(map_label)
    df_tr = df_tr.dropna(subset=["Y"]).copy()
    y = df_tr["Y"].astype(int).to_numpy()

    X_closed = closure(df_tr[comp_cols].fillna(0).to_numpy(dtype=float), 100.0)
    Z = clr(X_closed)

    # 分类
    metrics, importances = train_rf_cv(Z, y, cv=cv, seed=seed)
    imp = pd.DataFrame({"feature": comp_cols, "gini_importance": importances}).sort_values("gini_importance", ascending=False)
    pd.DataFrame([metrics]).to_csv(outdir / "rf_cv_metrics.csv", index=False, encoding="utf-8-sig")
    imp.to_csv(outdir / "rf_feature_importance.csv", index=False, encoding="utf-8-sig")

    # 混淆矩阵（基于CV预测再绘制）
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(RandomForestClassifier(n_estimators=600, random_state=seed, n_jobs=-1, class_weight="balanced_subsample"),
                               Z, y, cv=skf, method="predict")
    cm = confusion_matrix(y, y_pred, labels=[0,1])
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (RF, CLR)")
    plt.xticks([0,1], ["高钾(0)", "铅钡(1)"])
    plt.yticks([0,1], ["高钾(0)", "铅钡(1)"])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(outdir / "rf_cm.png", dpi=160)
    plt.close(fig)

    # 特征重要性
    fig = plt.figure(figsize=(6,4))
    plt.bar(imp["feature"], imp["gini_importance"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Gini importance")
    plt.title("RandomForest Feature Importance (CLR)")
    plt.tight_layout()
    fig.savefig(outdir / "rf_importance.png", dpi=160)
    plt.close(fig)

    # 亚类划分
    alias = { "SiO₂": "SiO2", "Al₂O₃":"Al2O3", "K₂O":"K2O", "CaO":"CaO", "MgO":"MgO", "PbO":"PbO", "BaO":"BaO" }
    comp_norm = df_tr[comp_cols].copy()
    comp_norm.columns = [alias.get(c,c) for c in comp_norm.columns]
    cols_used = list(comp_norm.columns)
    vars_pb = [c for c in ["PbO", "BaO", "SiO2", "Al2O3", "CaO"] if c in cols_used]
    vars_hk = [c for c in ["K2O", "SiO2", "Al2O3", "CaO", "MgO"] if c in cols_used]

    Xc_all = closure(comp_norm.to_numpy(float), 100.0)
    Zall = clr(Xc_all)

    # 构建列名索引
    col_idx = {c:i for i,c in enumerate(cols_used)}

    summary_lines = []

    for cls, name, featset in [(0, "高钾玻璃", vars_hk), (1, "铅钡玻璃", vars_pb)]:
        idx = np.where(y==cls)[0]
        if len(idx) < 6 or len(featset) < 2:
            continue
        Zsub = Zall[np.ix_(idx, [col_idx[c] for c in featset])]
        k_best, sc, labels, L = choose_k_by_silhouette(Zsub, kmin=2, kmax=5, method="ward")
        cophenetic_plot(L, outdir / f"dendrogram_{name}.png", f"{name}（Ward） — 最优k={k_best}, silhouette={sc:.3f}")
        lab_df = pd.DataFrame({"global_index": idx, "class": name, "subcluster": labels})
        lab_df.to_csv(outdir / f"subcluster_{name}.csv", index=False, encoding="utf-8-sig")

        X_closed_feats = closure(comp_norm.iloc[idx][featset].to_numpy(float), 100.0)
        fp = geo_mean_fingerprint(X_closed_feats, labels, featset)
        fp.to_csv(outdir / f"subcluster_{name}_fingerprint.csv", index=False, encoding="utf-8-sig")

        # Ward vs Average
        labels_avg, L_avg = hierarchical_cluster(Zsub, k=k_best, method="average")
        ari = adjusted_rand_score(labels, labels_avg)
        with open(outdir / f"sensitivity_{name}_linkage.txt", "w", encoding="utf-8") as f:
            f.write(f"Ward vs Average linkage ARI = {ari:.4f}\n")
        cophenetic_plot(L_avg, outdir / f"dendrogram_{name}_average.png", f"{name}（Average） — k={k_best}")

        # bootstrap 共聚
        co = bootstrap_coclustering(Zsub, k=k_best, B=boot, method="ward", seed=seed)
        np.savetxt(outdir / f"coclustering_{name}.csv", co, delimiter=",")
        # 簇内平均共聚概率
        stab_rows = []
        for c in range(1, k_best+1):
            members = np.where(labels == c)[0]
            if members.size >= 2:
                sub = co[np.ix_(members, members)]
                upp = sub[np.triu_indices_from(sub, k=1)]
                stab = float(upp.mean()) if upp.size > 0 else 1.0
            else:
                stab = 1.0
            stab_rows.append({"subcluster": c, "avg_coassignment": stab})
        pd.DataFrame(stab_rows).to_csv(outdir / f"stability_{name}.csv", index=False, encoding="utf-8-sig")

        summary_lines.append((name, k_best, sc, featset, ari))

    with open(outdir / "Q2_结果摘要.md", "w", encoding="utf-8") as f:
        f.write("# 问题二：分类规律与亚类划分（按思路文档实现）\n\n")
        f.write("## 监督分类（随机森林，CLR空间）\n\n")
        f.write(pd.DataFrame([metrics]).to_markdown(index=False))
        f.write("\n\n特征重要性见 `rf_feature_importance.csv` 与 `rf_importance.png`；混淆矩阵见 `rf_cm.png`。\n\n")
        f.write("## 亚类划分（层次聚类，Ward）\n\n")
        for (name, k_best, sc, feats, ari) in summary_lines:
            f.write(f"- **{name}**：最优亚类数 k={k_best}（silhouette={sc:.3f}），Ward vs Average 的 ARI = {ari:.3f}，使用变量：{', '.join(feats)}；\n")
            f.write(f"  - 指纹：见 `subcluster_{name}_fingerprint.csv`；标签：`subcluster_{name}.csv`；树状图：`dendrogram_{name}.png`。\n")
        f.write("\n## 敏感性分析\n\n- 联动法变更（Ward→Average）：见 `sensitivity_*_linkage.txt`，并对比 `dendrogram_*_average.png`。\n")
        f.write("- 自助法（bootstrap）：共聚频率矩阵 `coclustering_* .csv`；簇内平均共聚概率见 `stability_* .csv`。\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="C题")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--boot", type=int, default=100)
    args = parser.parse_args()
    main(args.base_dir, args.seed, args.cv, args.boot)
