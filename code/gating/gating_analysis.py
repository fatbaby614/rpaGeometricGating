"""
gating/gating_analysis.py — 路径 A / G2+G4a: ρ-gated transfer 主对照表
=====================================================================

输入:
    G1_subject_geometry_*.json  (被试级 ρ̄/Δacc/四臂精度, subject_geometry.py 产物)
    G3_threshold_calibration_*.json  (阈值/策略/LODO 标定结果, threshold_calibration.py 产物)

评估协议 (方案 8.2, 诚实性优先):
    - 主口径 = LODO gating: 每个数据集用 "另外两个数据集标定" 的策略评估 (零泄漏);
      pooled 主表 = 三个 held-out 被试向量拼接。
    - 若 G3 缺 LODO 条目 (单数据集冒烟), 回退 in-sample 策略并打 WARN (仅探索性)。
    - nested 5-fold 向量: 用与 G3 相同的 seed/permutation 重算, 并对账均值。
    - 检验复用 ric_da_core 的 paired_t_test(单侧 greater) + permutation_test(双侧),
      gating vs 各 always-臂 与 vs oracle (被试级四臂最优, 上界参照不可部署)。

图 (项目惯例: 最小 7pt, ≤4.5in 宽, png/pdf/svg, dpi=600):
    G_fig1_gating_vs_baselines  分组柱状 (每数据集 × 4基线+gating+oracle)
    G_fig2_rho_vs_gain          ρ̄–Δacc 散点 + Youden τ 线
    G_fig3_roc                  ρ̄→负迁移 ROC (分数据集 + pooled)

用法:
    python gating/gating_analysis.py                      # 自动取最新 G1/G3
    python gating/gating_analysis.py --g1 ... --g3 ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_HERE)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_curve, auc as sk_auc

from ric_da_core import paired_t_test, permutation_test
from threshold_calibration import (ARMS, RNG_SEED, BOOT_N,
                                   load_dataset_table, policy_acc, find_latest_g1)

PROJECT_ROOT = os.path.dirname(_CODE_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
})
DPI = 600
DATASET_COLORS = {"SEED": "#0F4D92", "SEED_IV": "#3775BA", "DEAP": "#42949E"}
ARM_COLORS = {"none": "#D8D8D8", "euclidean": "#B4C0E4",
              "riemann": "#7884B4", "rpa": "#0F4D92",
              "gating": "#C8562B", "oracle": "#767676"}


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_latest_g3() -> str:
    hits = sorted(glob.glob(os.path.join(RESULTS_DIR, "G3_threshold_calibration_*.json")))
    if not hits:
        raise FileNotFoundError("未找到 G3_threshold_calibration_*.json, 请先运行 threshold_calibration.py")
    return hits[-1]


def save_fig(fig, name: str):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        p = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    smallest = min(t.get_fontsize() for t in fig.findobj(mpl.text.Text))
    assert smallest >= 7.0, f"{name}: 存在 <7pt 文字 ({smallest:.1f}pt)"
    log(f"  [fig] {name} (min font {smallest:.1f}pt)")


def bootstrap_ci_mean(x: np.ndarray, rng) -> list:
    n = len(x)
    boots = np.array([x[rng.integers(0, n, n)].mean() for _ in range(BOOT_N)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [float(lo), float(hi)]


def subject_oracle(acc_stack: np.ndarray) -> np.ndarray:
    return acc_stack.max(axis=1)


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    sd = float(np.std(d, ddof=1))
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "delta": float(d.mean()),
        "dz": float(d.mean() / sd) if sd > 1e-12 else None,
        "t_test": paired_t_test(a, b, alternative="greater"),
        "permutation_test": permutation_test(a, b, n_permutations=10000),
    }


def main():
    ap = argparse.ArgumentParser(description="G2/G4a: 门控对照主表")
    ap.add_argument("--g1", default=None)
    ap.add_argument("--g3", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(RNG_SEED)

    g1_fp = args.g1 or find_latest_g1()
    g3_fp = args.g3 or find_latest_g3()
    log(f"[input] G1={g1_fp}")
    log(f"[input] G3={g3_fp}")
    with open(g1_fp) as f:
        g1 = json.load(f)
    with open(g3_fp) as f:
        g3 = json.load(f)
    ds_list = list(g1["geometry"].keys())
    tables = [load_dataset_table(g1, ds) for ds in ds_list]
    tmap = {t["ds"]: t for t in tables}

    honesty = g3.get("honesty", {})
    gate_ok = bool(honesty.get("gate_signal_pooled", False))
    log(f"[honesty] gate_signal_pooled={gate_ok} — {honesty.get('statement', 'N/A')}")

    # ── 1. held-out 门控向量 (LODO 主口径; 缺则 in-sample 回退) ──────
    gating_vec, mode_per_ds = {}, {}
    in_sample = g3["pooled"]["best_policy_in_sample"]
    for ds in ds_list:
        lodo = g3.get("lodo", {}).get(ds)
        if lodo is not None:
            pol, mode = lodo["policy"], "lodo"
        else:
            pol, mode = in_sample, "in_sample_fallback"
            log(f"[WARN] {ds}: G3 无 LODO 条目, 回退 in-sample 策略 (仅探索性, 不可宣称泛化)")
        t = tmap[ds]
        acc_stack = np.column_stack([t["acc"][a] for a in ARMS])
        g = policy_acc(t["rho"], acc_stack, pol["tau1"], pol["tau2"],
                       [ARMS.index(a) for a in pol["arms"]])
        gating_vec[ds] = g
        mode_per_ds[ds] = {"mode": mode, "policy": pol,
                           "gating_acc": float(g.mean())}
        log(f"  {ds}: [{mode}] gating={g.mean():.4f} arms={pol['arms']} "
            f"tau=({pol['tau1']},{pol['tau2']})")

    # pooled 向量 (被试序 = ds_list 拼接序)
    all_rho = np.concatenate([t["rho"] for t in tables])
    all_delta = np.concatenate([t["delta"] for t in tables])
    all_y = np.concatenate([t["y"] for t in tables])
    flat_acc = np.concatenate([np.column_stack([t["acc"][a] for a in ARMS])
                               for t in tables], axis=0)
    flat_gating = np.concatenate([gating_vec[ds] for ds in ds_list])
    flat_oracle = subject_oracle(flat_acc)

    # ── 2. nested 重算 + 对账 ────────────────────────────────────────
    nested_info = None
    n = len(all_rho)
    perm = np.random.default_rng(RNG_SEED).permutation(n)
    folds = np.array_split(perm, 5)
    if "folds" in g3.get("nested", {}):
        nested_g = np.empty(n)
        ok = True
        for k, fr in enumerate(g3["nested"]["folds"]):
            te = folds[k]
            pol = fr["policy"]
            nested_g[te] = policy_acc(all_rho[te], flat_acc[te], pol["tau1"],
                                      pol["tau2"], [ARMS.index(a) for a in pol["arms"]])
            if abs(float(nested_g[te].mean()) - fr["gating_acc"]) > 1e-9:
                ok = False
        if ok:
            nested_info = {"gating_acc": float(nested_g.mean()),
                           "always_rpa": float(flat_acc[:, ARMS.index("rpa")].mean()),
                           "always_none": float(flat_acc[:, ARMS.index("none")].mean())}
            log(f"  nested(对账通过): gating={nested_info['gating_acc']:.4f}")
        else:
            log("[WARN] nested permutation 对账失败, 跳过 (检查 G3/G1 输入是否同一)")

    # ── 3. 主表 + 检验 (pooled LODO 口径; per-dataset 描述统计) ──────
    out = {"meta": {"g1_file": os.path.basename(g1_fp), "g3_file": os.path.basename(g3_fp),
                    "timestamp": ts(), "protocol": "held-out gating: LODO (主), "
                            "nested5 交叉验证 (对账), oracle=被试级四臂上界",
                    "gate_signal_pooled": gate_ok,
                    "honesty_statement": honesty.get("statement")},
           "per_dataset": {}, "pooled": {}, "nested": nested_info,
           "policy_modes": mode_per_ds}

    for ds in ds_list:
        t = tmap[ds]
        acc_stack = np.column_stack([t["acc"][a] for a in ARMS])
        row = {"n": len(t["subjects"]),
               "arms": {a: {"mean": float(t["acc"][a].mean()),
                            "ci95": bootstrap_ci_mean(t["acc"][a], rng)} for a in ARMS},
               "gating": {"mean": float(gating_vec[ds].mean()),
                          "ci95": bootstrap_ci_mean(gating_vec[ds], rng),
                          "mode": mode_per_ds[ds]["mode"]},
               "oracle": {"mean": float(subject_oracle(acc_stack).mean())},
               "neg_rate": float(np.mean(t["y"]))}
        out["per_dataset"][ds] = row

    tests = {}
    for a in ARMS:
        tests[f"gating_vs_always_{a}"] = paired_stats(flat_gating, flat_acc[:, ARMS.index(a)])
    tests["gating_vs_oracle"] = paired_stats(flat_gating, flat_oracle)
    out["pooled"] = {
        "n": int(n),
        "gating_mean": float(flat_gating.mean()),
        "gating_ci95": bootstrap_ci_mean(flat_gating, rng),
        "oracle_mean": float(flat_oracle.mean()),
        "tests": tests,
    }
    for name, tv in tests.items():
        log(f"  pooled {name}: Δ={tv['delta']:+.4f} dz={tv['dz']} "
            f"p_t={tv['t_test']['p_value']:.4f} p_perm={tv['permutation_test']['p_value']:.4f}")

    # ── 4. 图 ─────────────────────────────────────────────────────────
    # Fig1: 分组柱状
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    groups = ["none", "euclidean", "riemann", "rpa", "gating", "oracle"]
    xw = np.arange(len(ds_list) + 1)  # +1 = pooled 组
    w = 0.13
    for gi, gname in enumerate(groups):
        means, los, his = [], [], []
        for ds in ds_list:
            r = out["per_dataset"][ds]
            v = (r["arms"][gname] if gname in ARMS else
                 (r["gating"] if gname == "gating" else r["oracle"]))
            means.append(v["mean"])
            if "ci95" in v:
                los.append(v["mean"] - v["ci95"][0]); his.append(v["ci95"][1] - v["mean"])
            else:
                los.append(0); his.append(0)
        pm = float(flat_gating.mean()) if gname == "gating" else (
            float(flat_oracle.mean()) if gname == "oracle" else float(flat_acc[:, ARMS.index(gname)].mean()))
        means.append(pm); los.append(0); his.append(0)
        ax.bar(xw + (gi - 2.5) * w, means, w, color=ARM_COLORS[gname],
               yerr=[los, his], capsize=1.5, error_kw={"lw": 0.6}, label=gname)
    ax.axvline(len(ds_list) - 0.5, color="#999999", lw=0.7, ls="--")
    ax.set_xticks(xw); ax.set_xticklabels(ds_list + ["pooled"])
    ax.set_ylabel("LSO accuracy (mean over held-out recording units)")
    ax.set_title("ρ-gated transfer vs fixed-alignment baselines"
                 + ("" if gate_ok else "  (exploratory: no pooled gate signal)"))
    lo = min([out["per_dataset"][d]["arms"]["none"]["mean"] for d in ds_list]
             + [float(flat_acc[:, 0].mean())])
    hi = max([out["per_dataset"][d]["oracle"]["mean"] for d in ds_list]
             + [float(flat_oracle.mean())])
    ax.set_ylim(lo - 0.02, hi + 0.035)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              handlelength=1.0, columnspacing=0.9)
    save_fig(fig, "G_fig1_gating_vs_baselines")
    plt.close(fig)

    # Fig2: ρ̄–Δacc 散点
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for ds in ds_list:
        t = tmap[ds]
        ax.scatter(t["rho"], t["delta"], s=8, alpha=0.75,
                   color=DATASET_COLORS.get(ds, "#666666"), label=ds, lw=0)
    yd = g3["pooled"]["rho"].get("youden")
    if yd:
        ax.axvline(yd["tau"], color="#B64342", lw=0.9, ls="--")
        ax.text(yd["tau"], ax.get_ylim()[1], f"  Youden τ={yd['tau']:.2f}",
                fontsize=7, color="#B64342", va="top")
    ax.axhline(0, color="#999999", lw=0.7)
    sp = g3["pooled"]["rho"]["spearman_vs_delta_acc"]
    au = g3["pooled"]["rho"].get("auc_negtransfer")
    ax.set_xlabel(r"$\bar{\rho}_m$ (cross-band mean)")
    ax.set_ylabel(r"$\Delta$acc (rpa − none)")
    ax.set_title(f"Spearman={sp['spearman']:+.2f} (p={sp['pvalue']:.2f})"
                 + (f", AUC={au:.2f}" if au is not None else ""))
    ax.legend(ncol=3, loc="lower right", handlelength=0.9)
    save_fig(fig, "G_fig2_rho_vs_gain")
    plt.close(fig)

    # Fig3: ROC
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    for ds in ds_list:
        t = tmap[ds]
        if len(np.unique(t["y"])) < 2:
            continue
        fpr, tpr, _ = roc_curve(t["y"], t["rho"])
        ax.plot(fpr, tpr, lw=1.0, color=DATASET_COLORS.get(ds, "#666666"),
                label=f"{ds} (AUC={sk_auc(fpr, tpr):.2f})")
    if len(np.unique(all_y)) >= 2:
        fpr, tpr, _ = roc_curve(all_y, all_rho)
        ax.plot(fpr, tpr, lw=1.4, color="black",
                label=f"pooled (AUC={sk_auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], lw=0.7, ls="--", color="#999999")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(r"ROC: $\bar{\rho}_m$ → fold-level negative transfer")
    ax.legend(loc="lower right", handlelength=1.2)
    save_fig(fig, "G_fig3_roc")
    plt.close(fig)

    fp = os.path.join(RESULTS_DIR, f"G4_gating_analysis_{ts()}.json")
    with open(fp, "w") as f:
        json.dump(out, f, indent=2, default=float)
    log(f"[saved] {fp}")


if __name__ == "__main__":
    main()
