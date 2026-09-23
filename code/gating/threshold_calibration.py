"""
gating/threshold_calibration.py — 路径 A / G3 (B1): 阈值统计校准
================================================================

输入: gating/subject_geometry.py 产出的 G1_subject_geometry_*.json
被试级门控变量 (跨带平均, 因 T1 分类是 5 带特征拼接的 fold 级评估):
    rho_bar_m = mean_band ρ̂_m      (被试级距离比, label-free)
    dci_bar_m = mean_band ΔCI_m    (RPA 对齐相对增益, label-free)
结果变量 (需要标签, 仅用于离线标定/验证):
    Δacc_m    = acc_rpa(m) − acc_none(m)
    y_m       = 1[Δacc_m < 0]      (fold 级负迁移)

协议 (方案 8.3):
    1. Spearman(ρ̄, Δacc) + bootstrap 95% CI (10,000), 分数据集 + pooled
    2. ROC-AUC(ρ̄→y), ROC-AUC(ΔCI→y), Youden's J 单阈值
    3. 区间策略网格: policy(ρ̄) 三区间 (低/中/高) → 4 臂任意组合 (64 种,
       含退化), τ 网格步长 0.05; 目标 = 标定池 mean acc 最大
    4. LODO 泛化: 2 数据集标定 → 第 3 个测试 (3 轮换全报)
    5. nested: pooled 被试级 5-fold CV (标定/评估 fold 不重叠)
    6. τ 稳定性 (跨轮换变异系数)

诚实性红线 (8.2): 若 Spearman 不显著 / AUC≈0.5, 输出负结果标记,
不硬凑阈值; gating 与 always-arm 的比较只报检验后结论。

用法:
    python gating/threshold_calibration.py                 # 自动取最新 G1 JSON
    python gating/threshold_calibration.py --g1 results/G1_subject_geometry_xxx.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")

ARMS = ["none", "euclidean", "riemann", "rpa"]
BOOT_N = 10_000
RNG_SEED = 20260923


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_latest_g1() -> str:
    hits = sorted(glob.glob(os.path.join(RESULTS_DIR, "G1_subject_geometry_*.json")))
    if not hits:
        raise FileNotFoundError("未找到 G1_subject_geometry_*.json, 请先运行 subject_geometry.py")
    return hits[-1]


def load_dataset_table(g1: dict, ds: str) -> dict:
    """把 G1 JSON 展平成被试级分析表 (rho_bar/dci_bar 跨带平均, 按 subjects 顺序)."""
    geom = g1["geometry"][ds]
    bands = list(geom.keys())
    subjects = geom[bands[0]]["subjects"]
    assert all(geom[b]["subjects"] == subjects for b in bands), f"{ds} 各 band subjects 顺序不一致"
    rho = np.mean([geom[b]["rho"] for b in bands], axis=0)
    dci = np.mean([geom[b]["dci_rpa"] for b in bands], axis=0)
    join = g1["t1_join"][ds]
    assert join["subjects"] == subjects, f"{ds} t1_join subjects 与 geometry 不一致"
    acc = {a: np.array(join["aligners"][a]["acc_per_subject"]) for a in ARMS}
    delta = np.array(join["delta_acc_rpa_minus_none"])
    y = np.array(join["neg_transfer_label"])
    assert np.allclose(delta, acc["rpa"] - acc["none"], atol=1e-12)
    return {"ds": ds, "subjects": subjects, "rho": rho, "dci": dci,
            "acc": acc, "delta": delta, "y": y}


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    rho0 = float(stats.spearmanr(x, y).statistic)
    n = len(x)
    boots = np.empty(BOOT_N)
    for b in range(BOOT_N):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) < 1e-12 or np.std(y[idx]) < 1e-12:
            boots[b] = np.nan
        else:
            boots[b] = stats.spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    p = float(stats.spearmanr(x, y).pvalue)
    return {"spearman": rho0, "pvalue": p, "ci95": [float(lo), float(hi)], "n": int(n)}


def youden_threshold(score: np.ndarray, y: np.ndarray) -> dict:
    """单阈值 (score > τ 判正类) 的 Youden's J 最优点."""
    cands = np.unique(score)
    best = None
    for t in cands:
        pred = score > t
        tpr = pred[y == 1].mean() if (y == 1).any() else 0.0
        fpr = pred[y == 0].mean() if (y == 0).any() else 0.0
        j = tpr - fpr
        if best is None or j > best["J"]:
            best = {"tau": float(t), "J": float(j), "tpr": float(tpr), "fpr": float(fpr)}
    return best


# ── 区间策略网格 ─────────────────────────────────────────────────────
TAUS = np.round(np.arange(0.25, 3.0 + 1e-9, 0.05), 2)
ARM_COMBOS = [(a, b, c) for a in range(4) for b in range(4) for c in range(4)]  # (低,中,高)


def policy_acc(rho: np.ndarray, acc_stack: np.ndarray, t1: float, t2: float,
               arms: tuple) -> np.ndarray:
    """按 ρ̄ 三区间 (≤t1, (t1,t2], >t2) 选臂, 返回每被试精度向量."""
    which = np.where(rho <= t1, 0, np.where(rho <= t2, 1, 2))
    arm_idx = np.array(arms)[which]
    return acc_stack[np.arange(len(rho)), arm_idx]


def search_best_policy(tables: list, rng: np.random.Generator) -> dict:
    """在给定被试集合 (tables 拼接) 上网格搜索最大化 mean acc 的策略."""
    rho = np.concatenate([t["rho"] for t in tables])
    acc_stack = np.concatenate([np.column_stack([t["acc"][a] for a in ARMS])
                                for t in tables], axis=0)
    lo, hi = float(np.min(rho)), float(np.max(rho))
    taus = TAUS[(TAUS >= lo - 0.5) & (TAUS <= hi + 0.5)]
    if len(taus) < 2:
        taus = np.array([lo - 1e-6, hi + 1e-6])
    best = None
    for t1 in taus:
        for t2 in taus:
            if t2 < t1:
                continue
            for arms in ARM_COMBOS:
                a = policy_acc(rho, acc_stack, t1, t2, arms)
                m = float(a.mean())
                if best is None or m > best["mean_acc"] + 1e-12:
                    best = {"tau1": float(t1), "tau2": float(t2),
                            "arms": [ARMS[k] for k in arms], "mean_acc": m}
    return best


def evaluate_policy(t: dict, tau1: float, tau2: float, arms: list) -> float:
    acc_stack = np.column_stack([t["acc"][a] for a in ARMS])
    return float(policy_acc(t["rho"], acc_stack, tau1, tau2,
                            [ARMS.index(a) for a in arms]).mean())


def main():
    ap = argparse.ArgumentParser(description="G3/B1: 阈值统计校准")
    ap.add_argument("--g1", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(RNG_SEED)

    g1_fp = args.g1 or find_latest_g1()
    log(f"[input] {g1_fp}")
    with open(g1_fp) as f:
        g1 = json.load(f)
    ds_list = list(g1["geometry"].keys())
    if len(ds_list) < 2:
        log(f"[WARN] 数据集数 {len(ds_list)} < 2, LODO/nested 无法进行, 仅报描述性统计")
    tables = [load_dataset_table(g1, ds) for ds in ds_list]

    out = {"meta": {"g1_file": os.path.basename(g1_fp), "timestamp": ts(),
                    "boot_n": BOOT_N, "seed": RNG_SEED, "tau_step": 0.05,
                    "gate_variable": "rho_bar / dci_bar = 跨 5 band 平均 (fold 级单值)"},
           "per_dataset": {}, "pooled": {}, "lodo": {}, "nested": {}, "honesty": {}}

    # ── 1+2: 相关性与判别力 ─────────────────────────────────────────
    all_rho = np.concatenate([t["rho"] for t in tables])
    all_dci = np.concatenate([t["dci"] for t in tables])
    all_delta = np.concatenate([t["delta"] for t in tables])
    all_y = np.concatenate([t["y"] for t in tables])

    def disc(score, delta, y, label):
        r = bootstrap_spearman(score, delta, rng)
        d = {"spearman_vs_delta_acc": r}
        if len(np.unique(y)) == 2:
            auc = float(roc_auc_score(y, score))
            d["auc_negtransfer"] = auc
            d["auc_note"] = ("正方向: 大ρ̄预测负迁移; AUC<0.5 时以 1-AUC 解读反向" if "rho" in label
                             else "更负ΔCI预测负迁移则 AUC<0.5")
            d["youden"] = youden_threshold(score, y)
        else:
            d["auc_negtransfer"] = None
            d["youden"] = None
            d["warning"] = f"{label}: y 单一类别, AUC/Youden 不可算"
        return d

    for t in tables:
        out["per_dataset"][t["ds"]] = {"rho": disc(t["rho"], t["delta"], t["y"], "rho"),
                                       "dci": disc(t["dci"], t["delta"], t["y"], "dci"),
                                       "n": len(t["subjects"]),
                                       "neg_rate": float(np.mean(t["y"]))}
        log(f"  {t['ds']}: n={len(t['subjects'])} neg_rate={np.mean(t['y']):.3f} "
            f"spearman(rho)={out['per_dataset'][t['ds']]['rho']['spearman_vs_delta_acc']['spearman']:.3f} "
            f"auc(rho)={out['per_dataset'][t['ds']]['rho'].get('auc_negtransfer')}")
    out["pooled"] = {"rho": disc(all_rho, all_delta, all_y, "rho"),
                     "dci": disc(all_dci, all_delta, all_y, "dci"),
                     "n": int(len(all_y)), "neg_rate": float(np.mean(all_y))}
    log(f"  pooled: n={len(all_y)} spearman(rho)="
        f"{out['pooled']['rho']['spearman_vs_delta_acc']['spearman']:.3f} "
        f"auc(rho)={out['pooled']['rho'].get('auc_negtransfer')}")

    # ── 诚实性门 (8.2 红线): pooled 不显著则标记, 后续结果仅作探索 ──
    sp = out["pooled"]["rho"]["spearman_vs_delta_acc"]
    auc_r = out["pooled"]["rho"].get("auc_negtransfer")
    gate_ok = (sp["pvalue"] < 0.05) or (auc_r is not None and
                                        max(auc_r, 1 - auc_r) >= 0.6)
    out["honesty"]["gate_signal_pooled"] = bool(gate_ok)
    out["honesty"]["statement"] = ("pooled 门控信号存在 (继续标定)" if gate_ok else
                                   "pooled Spearman 不显著且 AUC<0.6: "
                                   "几何门控为负结果, 主表只报 LODO 泛化, 不宣称实用增益")

    # ── 3: pooled 最优策略 + 各基线 ─────────────────────────────────
    pooled_tables = tables
    best = search_best_policy(pooled_tables, rng)
    baselines = {f"always_{a}": float(np.concatenate([t["acc"][a] for t in tables]).mean())
                 for a in ARMS}
    oracle = float(np.max(np.concatenate(
        [np.column_stack([t["acc"][a] for a in ARMS]) for t in tables], axis=0), axis=1).mean())
    out["pooled"]["best_policy_in_sample"] = best
    out["pooled"]["baselines"] = baselines
    out["pooled"]["oracle_upper_bound"] = oracle
    log(f"  in-sample best: {best}")

    # ── 4: LODO ─────────────────────────────────────────────────────
    lodo_rows = []
    for held in ds_list:
        train = [t for t in tables if t["ds"] != held]
        test = [t for t in tables if t["ds"] == held][0]
        if not train:
            continue
        pol = search_best_policy(train, rng)
        g_acc = evaluate_policy(test, pol["tau1"], pol["tau2"], pol["arms"])
        row = {"held_out": held, "n_test": len(test["subjects"]),
               "policy": pol, "gating_acc_on_heldout": g_acc,
               "baselines_on_heldout": {f"always_{a}": float(test["acc"][a].mean())
                                        for a in ARMS}}
        row["delta_vs_always_rpa"] = g_acc - row["baselines_on_heldout"]["always_rpa"]
        row["delta_vs_always_none"] = g_acc - row["baselines_on_heldout"]["always_none"]
        lodo_rows.append(row)
        out["lodo"][held] = row
        log(f"  LODO[-{held}]: gating={g_acc:.4f} vs rpa={row['baselines_on_heldout']['always_rpa']:.4f} "
            f"vs none={row['baselines_on_heldout']['always_none']:.4f} (policy={pol['arms']})")
    if len(lodo_rows) >= 2:
        taus1 = [r["policy"]["tau1"] for r in lodo_rows]
        taus2 = [r["policy"]["tau2"] for r in lodo_rows]
        out["lodo"]["tau_stability"] = {
            "tau1_list": taus1, "tau2_list": taus2,
            "tau1_cv": float(np.std(taus1) / max(np.mean(taus1), 1e-9)),
            "tau2_cv": float(np.std(taus2) / max(np.mean(taus2), 1e-9)),
        }

    # ── 5: nested 5-fold (pooled, 被试级) ───────────────────────────
    idx_all = np.concatenate([[i] * len(t["subjects"]) for i, t in enumerate(tables)])
    flat_rho = all_rho
    flat_acc = np.concatenate([np.column_stack([t["acc"][a] for a in ARMS])
                               for t in tables], axis=0)
    flat_delta = all_delta
    n = len(flat_rho)
    perm = np.random.default_rng(RNG_SEED).permutation(n)
    folds = np.array_split(perm, 5)
    nested_rows = []
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        ttr = {"ds": f"nested_train_fold{k}", "rho": flat_rho[tr],
               "acc": {a: flat_acc[tr, ARMS.index(a)] for a in ARMS}}
        pol = search_best_policy([ttr], rng)
        g = float(policy_acc(flat_rho[te], flat_acc[te], pol["tau1"], pol["tau2"],
                             [ARMS.index(a) for a in pol["arms"]]).mean())
        nested_rows.append({"fold": k, "policy": pol, "gating_acc": g,
                            "always_rpa": float(flat_acc[te, ARMS.index("rpa")].mean()),
                            "always_none": float(flat_acc[te, ARMS.index("none")].mean())})
    out["nested"] = {"folds": nested_rows,
                     "gating_mean": float(np.mean([r["gating_acc"] for r in nested_rows])),
                     "always_rpa_mean": float(np.mean([r["always_rpa"] for r in nested_rows])),
                     "always_none_mean": float(np.mean([r["always_none"] for r in nested_rows]))}
    log(f"  nested5: gating={out['nested']['gating_mean']:.4f} "
        f"rpa={out['nested']['always_rpa_mean']:.4f} none={out['nested']['always_none_mean']:.4f}")

    fp = os.path.join(RESULTS_DIR, f"G3_threshold_calibration_{ts()}.json")
    with open(fp, "w") as f:
        json.dump(out, f, indent=2, default=float)
    log(f"[saved] {fp}")


if __name__ == "__main__":
    main()
