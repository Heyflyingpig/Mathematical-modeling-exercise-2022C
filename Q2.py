#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2_hierRF.py — 2022C 题 问题二：分类规律 + 亚类划分 + 敏感性分析（稳健修正版）
---------------------------------------------------------------------------------
改进要点：
1) 列名规范化与合并：统一 “SiO₂/SiO2(%) / wt.% / 空格 / 下标数字 / 全角符号”等写法，并对规范化后重复列求和。
2) 亚类特征集兜底：若推荐特征缺失(<min_feats)，自动用该类样本的“CLR 方差 Top-k 成分”（默认 k=5）。
3) 样本量自适应：各类样本 n<6 时仍尝试（n>=4），并把层次聚类 kmax 截断为 min(5, n-1)。
4) 全流程诊断：把关键信息写到 outputs_q2/diagnostics.json，方便核对数据格式。
"""

import argparse, json, math
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (confusion_matrix, roc_auc_score, accuracy_score,
                             f1_score, precision_score, recall_score, silhouette_score,
                             adjusted_rand_score)
from sklearn.ensemble import RandomForestClassifier

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
import matplotlib.pyplot as plt
import preprocess

# ============================== 工具函数 ==============================

SUB_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def canon_name(s: str) -> str:
    """将列名规范化到标准氧化物写法：
       - 去空格、去 %、去 ( % ) / （%）
       - 下标数字转正常数字（₂→2 等）
       - 全角符号转半角
       - 常见单位词去除( wt.% / mass% 等 )
    """
    x = str(s).strip()
    x = x.translate(SUB_MAP)                  # 下标数字 → 正常数字
    x = x.replace("％", "%")
    x = x.replace("（", "(").replace("）", ")")
    x = x.replace(" ", "")
    # 去单位/百分号
    for t in ["(%)", "%", "wt.%", "wt%", "mass%", "质量分数", "含量"]:
        x = x.replace(t, "")
    # 一些常见别名统一
    aliases = {
        "SiO₂":"SiO2","Al₂O₃":"Al2O3","K₂O":"K2O","Na₂O":"Na2O","CaO":"CaO","MgO":"MgO",
        "Fe₂O₃":"Fe2O3","TiO₂":"TiO2","SO₃":"SO3","P₂O₅":"P2O5","MnO₂":"MnO2"
    }
    x = aliases.get(x, x)
    return x

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

def inv_clr(Z: np.ndarray, total: float = 100.0) -> np.ndarray:
    X = np.exp(Z)
    return closure(X, total)

def detect_comp_cols(df: pd.DataFrame) -> List[str]:
    """启发式识别成分列（包含氧化物形态），随后再统一规范化"""
    cols = []
    for c in df.columns:
        s = str(c)
        if "O" in s and any(ch.isdigit() for ch in s):  # 如 SiO2, Al2O3, Fe2O3...
            cols.append(c)
        elif any(k in s for k in ["SiO","Al2O","K2O","Na2O","CaO","MgO","PbO","BaO","Fe2O","SrO","TiO","MnO","CuO","ZnO","SnO","Sb2O","P2O","SO","Cl"]):
            cols.append(c)
    # 去重保持顺序
    out, seen = [], set()
    for c in cols:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def hierarchical_cluster(Z: np.ndarray, k: int, method: str = "ward"):
    D = pdist(Z, metric="euclidean")
    L = linkage(D, method=method)
    labels = fcluster(L, t=k, criterion="maxclust")
    return labels, L

def choose_k_by_silhouette(Z: np.ndarray, kmin=2, kmax=5, method="ward"):
    kmax = max(2, min(kmax, Z.shape[0]-1))  # 保证 kmax 合理
    best_k, best_score, best_labels, best_link = None, -1.0, None, None
    for k in range(2, kmax+1):
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

# ============================== 主流程 ==============================

def main(base_dir: str, seed: int, cv: int, boot: int, min_feats: int, auto_topk: int, min_per_class: int):
    outdir = Path("outputs_q2"); outdir.mkdir(parents=True, exist_ok=True)
    diag = {}

    # ---------- 读取数据 ----------
    df1 = preprocess.df1
    df2 = preprocess.df2

    # ---------- 编号/类型/风化列 ----------
    id_col = next((c for c in df1.columns if "编号" in str(c)), None)
    type_col = next((c for c in df1.columns if ("类型" in str(c) or "类别" in str(c))), None)
    weather_col = next((c for c in df1.columns if "风化" in str(c)), None)
    if id_col is None or type_col is None:
        raise ValueError("表单1中未找到‘编号’或‘类型/类别’列")

    id_col2 = id_col if id_col in df2.columns else next((c for c in df2.columns if "编号" in str(c)), None)
    if id_col2 is None:
        raise ValueError("表单2无法匹配编号列")

    meta_cols = [id_col, type_col] + ([weather_col] if weather_col else [])
    meta = df1[meta_cols].copy()
    meta = meta.rename(columns={id_col:"ID", type_col:"TYPE", (weather_col or "WEATHER"):"WEATHER"})
    if not weather_col:
        meta["WEATHER"] = np.nan

    data = df2.copy()
    if id_col2 != "ID":
        data = data.rename(columns={id_col2:"ID"})
    df = data.merge(meta, on="ID", how="left")

    # ---------- 成分列识别 + 规范化 + 合并 ----------
    comp_raw = detect_comp_cols(df)
    if not comp_raw:
        raise ValueError("未识别到成分列")
    # 规范化名到 canon，并聚合同名列（求和）
    canon_map = {c: canon_name(c) for c in comp_raw}
    df_comp = df[comp_raw].copy()
    df_comp.columns = [canon_map[c] for c in comp_raw]
    df_comp = df_comp.groupby(axis=1, level=0).sum()  # 合并同名列
    comp_cols = list(df_comp.columns)

    # 有效样本筛选
    s = df_comp.fillna(0).sum(axis=1)
    mask_valid = (s >= 85.0) & (s <= 105.0)
    df = df.loc[mask_valid].reset_index(drop=True)
    df_comp = df_comp.loc[mask_valid].reset_index(drop=True)

    # ---------- 风化筛选（训练集仅无风化；若无该列或取值不匹配，则不过滤） ----------
    if "WEATHER" in df.columns:
        weather_str = set(df["WEATHER"].astype(str))
        non_weather_tokens = {"无风化","未风化","否","无","0","0.0","nan","NaN"}
        keep_vals = [v for v in weather_str if v in non_weather_tokens]
        df_tr = df.copy() if len(keep_vals)==0 else df[df["WEATHER"].astype(str).isin(keep_vals)].copy()
        df_comp_tr = df_comp.loc[df_tr.index].reset_index(drop=True)
        df_tr = df_tr.reset_index(drop=True)
    else:
        df_tr, df_comp_tr = df.copy(), df_comp.copy()

    # ---------- 标签 ----------
    def map_label(s):
        s = str(s)
        if ("铅" in s) and ("钡" in s): return 1
        if ("高钾" in s) or ("钾" in s): return 0
        return np.nan
    df_tr["Y"] = df_tr["TYPE"].map(map_label)
    df_tr = df_tr.dropna(subset=["Y"]).reset_index(drop=True)
    df_comp_tr = df_comp_tr.loc[df_tr.index].reset_index(drop=True)
    y = df_tr["Y"].astype(int).to_numpy()

    # ---------- CLR 特征 ----------
    X_closed = closure(df_comp_tr.fillna(0).to_numpy(float), 100.0)
    Z = clr(X_closed)

    # ---------- 监督分类（RF + CV） ----------
    skf = StratifiedKFold(n_splits=min(cv, np.unique(y, return_counts=True)[1].min()), shuffle=True, random_state=seed)
    rf = RandomForestClassifier(n_estimators=800, random_state=seed, n_jobs=-1, class_weight="balanced_subsample")
    y_pred = cross_val_predict(rf, Z, y, cv=skf, method="predict")
    y_prob = cross_val_predict(rf, Z, y, cv=skf, method="predict_proba")[:,1]
    rf.fit(Z, y)

    metrics = dict(
        acc = float(accuracy_score(y, y_pred)),
        prec = float(precision_score(y, y_pred)),
        rec = float(recall_score(y, y_pred)),
        f1 = float(f1_score(y, y_pred)),
        auc = float(roc_auc_score(y, y_prob)) if len(np.unique(y))==2 else float("nan")
    )
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({"feature": comp_cols, "gini_importance": importances}).sort_values("gini_importance", ascending=False)

    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(outdir / "rf_cv_metrics.csv", index=False, encoding="utf-8-sig")
    imp_df.to_csv(outdir / "rf_feature_importance.csv", index=False, encoding="utf-8-sig")

    # 混淆矩阵图
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
    fig.savefig(outdir / "rf_cm.png", dpi=160); plt.close(fig)

    # 特征重要性图
    fig = plt.figure(figsize=(6,4))
    plt.bar(imp_df["feature"], imp_df["gini_importance"])
    plt.xticks(rotation=60, ha="right"); plt.ylabel("Gini importance")
    plt.title("RandomForest Feature Importance (CLR)")
    plt.tight_layout()
    fig.savefig(outdir / "rf_importance.png", dpi=160); plt.close(fig)

    # ---------- 亚类划分：两大类分别 ----------
    # 推荐特征（规范化后名称）
    rec_pb = ["PbO","BaO","SiO2","Al2O3","CaO"]
    rec_hk = ["K2O","SiO2","Al2O3","CaO","MgO"]

    # 记录诊断信息
    diag["comp_cols_after_canon"] = comp_cols
    diag["n_total_after_valid"] = int(df_tr.shape[0])
    diag["class_counts"] = { "高钾(0)": int((y==0).sum()), "铅钡(1)": int((y==1).sum()) }

    summary_lines = []

    for cls, name, rec in [(0,"高钾玻璃", rec_hk), (1,"铅钡玻璃", rec_pb)]:
        idx = np.where(y==cls)[0]
        n = int(idx.size)
        if n < max(4, min_per_class):
            diag[f"{name}_skip_reason"] = f"样本数不足 n={n} (<{max(4, min_per_class)})"
            continue

        # 推荐特征可用性
        avail = [c for c in rec if c in comp_cols]
        use_feats = avail.copy()

        # 若不足 min_feats，则自动选择该类样本方差 Top-k
        if len(use_feats) < min_feats:
            Z_class = Z[idx]
            var = np.var(Z_class, axis=0)
            top_idx = np.argsort(var)[::-1][:auto_topk]
            auto_feats = [comp_cols[i] for i in top_idx]
            # 避免重复，把缺口补齐
            for f in auto_feats:
                if f not in use_feats:
                    use_feats.append(f)
            use_feats = use_feats[:max(min_feats, min(auto_topk, len(comp_cols)))]

        # 记录诊断
        diag[f"{name}_use_feats"] = use_feats
        diag[f"{name}_n"] = n

        # 构造子矩阵（使用闭合后的真实比例求指纹；聚类用 CLR）
        Zsub = Z[idx][:, [comp_cols.index(f) for f in use_feats]]
        X_closed_feats = closure(df_comp_tr.iloc[idx][use_feats].to_numpy(float), 100.0)

        # 选 k
        k_best, sc, labels, L = choose_k_by_silhouette(Zsub, kmin=2, kmax=5, method="ward")
        if k_best is None:
            diag[f"{name}_skip_reason"] = "无法确定有效的 k"
            continue

        # 保存标签与树状图
        lab_df = pd.DataFrame({"global_index": idx, "class": name, "subcluster": labels})
        lab_df.to_csv(outdir / f"subcluster_{name}.csv", index=False, encoding="utf-8-sig")

        fig = plt.figure(figsize=(8,4))
        dendrogram(L, no_labels=True)
        plt.title(f"{name}（Ward） — 最优k={k_best}, silhouette={sc:.3f}")
        plt.tight_layout(); fig.savefig(outdir / f"dendrogram_{name}.png", dpi=150); plt.close(fig)

        # 指纹（几何均值，逆CLR）
        fp = geo_mean_fingerprint(X_closed_feats, labels, use_feats)
        fp.to_csv(outdir / f"subcluster_{name}_fingerprint.csv", index=False, encoding="utf-8-sig")

        # 敏感性：Ward vs Average（ARI）
        labels_avg, L_avg = hierarchical_cluster(Zsub, k=k_best, method="average")
        ari = adjusted_rand_score(labels, labels_avg)
        with open(outdir / f"sensitivity_{name}_linkage.txt", "w", encoding="utf-8") as f:
            f.write(f"Ward vs Average linkage ARI = {ari:.4f}\n")
        fig = plt.figure(figsize=(8,4))
        dendrogram(L_avg, no_labels=True)
        plt.title(f"{name}（Average） — k={k_best}")
        plt.tight_layout(); fig.savefig(outdir / f"dendrogram_{name}_average.png", dpi=150); plt.close(fig)

        # 自助法共聚频率
        rng = np.random.RandomState(seed)
        n_sub = Zsub.shape[0]
        co = np.zeros((n_sub, n_sub), dtype=float)
        B = max(boot, 0)
        for b in range(B):
            ridx = rng.randint(0, n_sub, size=n_sub)
            uniq, _ = np.unique(ridx, return_inverse=True)
            Zb = Zsub[uniq]
            labs_b, _ = hierarchical_cluster(Zb, k=k_best, method="ward")
            for a in range(len(uniq)):
                for bb in range(a+1, len(uniq)):
                    if labs_b[a] == labs_b[bb]:
                        ia, ib = uniq[a], uniq[bb]
                        co[ia, ib] += 1.0; co[ib, ia] += 1.0
        if B > 0:
            co /= float(B); np.fill_diagonal(co, 1.0)
            np.savetxt(outdir / f"coclustering_{name}.csv", co, delimiter=",")

            # 簇内平均共聚概率
            stab_rows = []
            for c in range(1, k_best+1):
                members = np.where(labels == c)[0]
                if members.size >= 2:
                    sub = co[np.ix_(members, members)]
                    upp = sub[np.triu_indices_from(sub, k=1)]
                    stab = float(upp.mean()) if upp.size>0 else 1.0
                else:
                    stab = 1.0
                stab_rows.append({"subcluster": c, "avg_coassignment": stab})
            pd.DataFrame(stab_rows).to_csv(outdir / f"stability_{name}.csv", index=False, encoding="utf-8-sig")

        summary_lines.append((name, k_best, sc, use_feats, ari))

    # ---------- 摘要 ----------
    with open(outdir / "Q2_结果摘要.md", "w", encoding="utf-8") as f:
        f.write("# 问题二：分类规律与亚类划分（稳健修正版）\n\n")
        f.write("## 监督分类（随机森林，CLR 空间）\n\n")
        f.write(pd.DataFrame([metrics]).to_markdown(index=False))
        f.write("\n\n特征重要性：`rf_feature_importance.csv`；混淆矩阵：`rf_cm.png`。\n\n")
        f.write("## 亚类划分（层次聚类，Ward）\n\n")
        if summary_lines:
            for (name, k_best, sc, feats, ari) in summary_lines:
                f.write(f"- **{name}**：最优亚类数 k={k_best}（silhouette={sc:.3f}），Ward↔Average 的 ARI={ari:.3f}；使用变量：{', '.join(feats)}。\n")
                f.write(f"  指纹：`subcluster_{name}_fingerprint.csv`；标签：`subcluster_{name}.csv`；树状图：`dendrogram_{name}.png`。\n")
        else:
            f.write("- 注意：本次运行未生成亚类结果。请检查 `outputs_q2/diagnostics.json` 中的样本量与可用特征，或放宽 `--min_per_class/--min_feats`。\n")

    # ---------- 诊断信息 ----------
    diag["metrics"] = metrics
    diag["summary_lines_empty"] = (len(summary_lines)==0)
    with open(outdir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)

# ============================== 入口 ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="C题")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--boot", type=int, default=100, help="bootstrap 次数（0 关闭）")
    parser.add_argument("--min_feats", type=int, default=3, help="推荐特征不足时的最小特征数阈值")
    parser.add_argument("--auto_topk", type=int, default=5, help="方差兜底挑选的特征个数上限")
    parser.add_argument("--min_per_class", type=int, default=4, help="每类参与聚类的最小样本数")
    args = parser.parse_args()
    main(args.base_dir, args.seed, args.cv, args.boot, args.min_feats, args.auto_topk, args.min_per_class)
