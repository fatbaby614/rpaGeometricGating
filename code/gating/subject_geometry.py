"""
gating/subject_geometry.py — 路径 A / G1: 被试级几何量 (ρ̂, ΔCI)
================================================================

为 ρ-gated transfer (创新路径 A1) 计算每个 LOSO fold (=测试被试) 的
无标签几何量, 供后续 threshold_calibration.py / gating_analysis.py 使用。

产出 (每个数据集 × 频段):
    ρ̂_m   : 被试级距离比 = d_F(M_m, center_-m) / mean_{s≠m} d_F(M_s, center_-m)
            (center_-m 为留一源池代表点的黎曼均值; label-free)
    ΔCI_m : RPA 对齐前后, 被试 m 到源池平均距离的相对变化 (负 = 拉近)
    join  : 与 T1 JSON 的 acc_per_subject 按 subjects 顺序对齐,
            附 Δacc(rpa-none) 与负迁移标签 y_m

口径对齐 (方案 8.1, 与 frechet_ci.py / run_T7 逐字一致):
    - 代表点: mean_riemann(covs[s][band][:30], maxiter=30) + 对称化
    - 距离:   pyriemann distance(metric='riemann') (AIRM Fréchet)
    - 聚合:   上三角/行均, NaN 防护同 frechet_ci

三重自检 (方案 R2/R4):
    1. 自算代表点的 D 矩阵 == compute_subject_frechet_matrix 输出 (allclose)
    2. 聚合回数据集级 CI == d1_T7_frechet_ci_*.json 存档值 (rtol)
    3. T1 acc_per_subject 均值 == 存档 acc_mean, 且长度 == len(subjects)

用法:
    python gating/subject_geometry.py                       # 3 数据集 32ch 5band
    python gating/subject_geometry.py --datasets SEED       # 冒烟
    python gating/subject_geometry.py --channels 16ch       # G4b 扩展轮
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

# 路径修正: 本文件在 code/gating/, 上一级即 code/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ric_da_core import (
    EuclideanAlignment, RiemannianAlignment, RiemannianProcrustesAlignment,
    BANDS_5, BANDS_8,
)
from frechet_ci import compute_subject_frechet_matrix, frechet_distance
from pyriemann.utils.mean import mean_riemann
from loaders.seed_loader import precompute_all_seed, list_subjects as list_seed_subs
from loaders.seed_iv_loader import precompute_all_seed_iv, list_subjects as list_seed_iv_subs
from loaders.deap_loader import precompute_all_deap, list_subjects as list_deap_subs

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_LOADERS = {
    "SEED":    (list_seed_subs,    precompute_all_seed),
    "SEED_IV": (list_seed_iv_subs, precompute_all_seed_iv),
    "DEAP":    (list_deap_subs,    precompute_all_deap),
}

# 门控动作空间 (label-free 四臂); ΔCI 用 rpa 臂, 与 T7 CI 口径一致
CI_ALIGNERS = {
    "euclidean": EuclideanAlignment,
    "riemann":   RiemannianAlignment,
    "rpa":       RiemannianProcrustesAlignment,
}

MEAN_MAXITER = 30   # 同 frechet_ci.compute_subject_frechet_matrix
MAX_EPOCHS = 30     # 同 T7 默认 max_epochs


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def band_key(band: Tuple[float, float]) -> str:
    return f"{band[0]}-{band[1]}Hz"


def subject_representatives(covs: Dict, subjects: List[str], band,
                            max_epochs: int = MAX_EPOCHS) -> Dict[str, np.ndarray]:
    """每被试代表点: mean_riemann(covs[:max_epochs], maxiter=30) + 对称化 (T7 口径)."""
    M = {}
    for s in subjects:
        covs_s = covs[s][band][:max_epochs]
        if covs_s.shape[0] == 0:
            raise ValueError(f"被试 {s} band={band} covs 为空, 无法计算代表点")
        m = mean_riemann(covs_s, maxiter=MEAN_MAXITER)
        M[s] = (m + m.T) / 2
        if not np.all(np.isfinite(M[s])):
            raise ValueError(f"被试 {s} band={band} 代表点含 NaN/Inf (mean_riemann 不收敛)")
    return M


def frechet_matrix_from_means(M: Dict[str, np.ndarray], subjects: List[str]) -> np.ndarray:
    """由代表点字典算成对 Fréchet 距离矩阵 (与 frechet_ci 同距离函数)."""
    n = len(subjects)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = frechet_distance(M[subjects[i]], M[subjects[j]])
            D[i, j] = d
            D[j, i] = d
    return D


def rho_leave_one_out(M: Dict[str, np.ndarray], subjects: List[str]) -> Dict[str, dict]:
    """被试级 ρ̂: 留一源池中心距离 / 源池到该中心平均距离 (label-free)."""
    out = {}
    for i, m in enumerate(subjects):
        pool = [s for j, s in enumerate(subjects) if j != i]
        center = mean_riemann(np.stack([M[s] for s in pool]), maxiter=MEAN_MAXITER)
        center = (center + center.T) / 2
        if not np.all(np.isfinite(center)):
            raise ValueError(f"留一池中心含 NaN/Inf (排除 {m})")
        d_out = frechet_distance(M[m], center)
        d_ref_vals = [frechet_distance(M[s], center) for s in pool]
        d_ref = float(np.mean(d_ref_vals))
        out[m] = {
            "d_out": float(d_out),
            "d_ref": d_ref,
            "rho": float(d_out / d_ref) if d_ref > 1e-10 else float("nan"),
        }
    return out


def row_mean_offdiag(D: np.ndarray) -> np.ndarray:
    """每行去掉对角后的均值 (被试 m 到其余被试的平均距离)."""
    n = D.shape[0]
    out = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        out[i] = D[i, mask].mean()
    return out


def find_latest(pattern: str) -> str:
    hits = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    if not hits:
        raise FileNotFoundError(f"未找到结果文件: {pattern} (目录 {RESULTS_DIR})")
    return hits[-1]


def load_saved_t7() -> dict:
    fp = find_latest("d1_T7_frechet_ci_*.json")
    log(f"  [自检] T7 存档: {os.path.basename(fp)}")
    with open(fp) as f:
        return json.load(f)


def load_saved_t1() -> dict:
    fp = find_latest("d1_T1_baseline_*.json")
    log(f"  [自检] T1 存档: {os.path.basename(fp)}")
    with open(fp) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="G1: 被试级几何量 (ρ̂, ΔCI)")
    ap.add_argument("--datasets", default="SEED,SEED_IV,DEAP")
    ap.add_argument("--bands", default="5band", choices=["5band", "8band"])
    ap.add_argument("--channels", default="32ch")
    ap.add_argument("--ci-rtol", type=float, default=1e-6,
                    help="聚合 CI 与 T7 存档对比的相对容差")
    args = ap.parse_args()

    freq_bands = BANDS_5 if args.bands == "5band" else BANDS_8
    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    t0 = time.time()

    t7_saved = load_saved_t7()
    t1_saved = load_saved_t1()

    results = {
        "meta": {
            "script": "gating/subject_geometry.py",
            "timestamp": ts(),
            "bands": args.bands,
            "channels": args.channels,
            "max_epochs": MAX_EPOCHS,
            "mean_maxiter": MEAN_MAXITER,
            "fold_order_basis": "list_subjects() 确定性顺序 (与 evaluate_loso_classification 的 fold 顺序同源)",
            "definitions": {
                "rho": "d_out/d_ref; d_out=δ_F(M_m, mean_riemann(留一池代表点)); d_ref=mean_{s≠m} δ_F(M_s, 同中心)",
                "dci": "(mean_after_row - mean_before_row)/mean_before_row, after=RPA fit_transform(全体池, T7口径)",
            },
        },
        "geometry": {},
        "t1_join": {},
        "selfcheck": {"d_matrix_vs_frechet_ci": {}, "ci_vs_t7_saved": {}, "t1_acc_recompute": {}},
    }

    for ds in ds_list:
        list_subs, precompute = DATASET_LOADERS[ds]
        subjects = list_subs()
        log(f"== {ds}: {len(subjects)} subjects, channels={args.channels}, bands={args.bands}")
        covs, _ = precompute(subjects, bands=args.bands, channels=args.channels)
        missing = [s for s in subjects if s not in covs]
        if missing:
            raise RuntimeError(f"{ds} covs 缺失被试 (loader 跳空?): {missing}")

        band_covs_all = {s: covs[s] for s in subjects}
        results["geometry"][ds] = {}
        results["t1_join"][ds] = {"subjects": list(subjects), "aligners": {}}

        # ── T1 join 自检 (R2): 长度 + 均值复现 ────────────────────────
        for al in ["none", "euclidean", "riemann", "rpa"]:
            leaf = t1_saved[ds][al]
            aps = leaf["acc_per_subject"]
            if len(aps) != len(subjects):
                raise RuntimeError(
                    f"R2 失败: {ds}/{al} acc_per_subject 长度 {len(aps)} != subjects {len(subjects)}")
            recomputed = float(np.mean(aps))
            saved_mean = float(leaf["acc_mean"])
            ok = abs(recomputed - saved_mean) < 1e-9
            results["selfcheck"]["t1_acc_recompute"][f"{ds}/{al}"] = {
                "saved_acc_mean": saved_mean,
                "recomputed_acc_mean": recomputed,
                "abs_diff": abs(recomputed - saved_mean),
                "ok": bool(ok),
            }
            if not ok:
                raise RuntimeError(f"R2 失败: {ds}/{al} acc_mean 无法复现")
            results["t1_join"][ds]["aligners"][al] = {
                "acc_per_subject": [float(a) for a in aps],
            }

        # Δacc 与负迁移标签 (rpa - none)
        acc_rpa = np.array(t1_saved[ds]["rpa"]["acc_per_subject"], dtype=float)
        acc_none = np.array(t1_saved[ds]["none"]["acc_per_subject"], dtype=float)
        results["t1_join"][ds]["delta_acc_rpa_minus_none"] = (acc_rpa - acc_none).tolist()
        results["t1_join"][ds]["neg_transfer_label"] = ((acc_rpa - acc_none) < 0).astype(int).tolist()

        for band in freq_bands:
            bk = band_key(band)
            log(f"  band {bk} ...")

            # 1) 代表点 + 自算 D 矩阵, 与 frechet_ci 输出交叉断言 (R4-a)
            M = subject_representatives(covs, subjects, band)
            D_own = frechet_matrix_from_means(M, subjects)
            D_ref_matrix = compute_subject_frechet_matrix(covs, subjects, band, MAX_EPOCHS)
            d_ok = bool(np.allclose(D_own, D_ref_matrix, rtol=1e-10, atol=1e-10, equal_nan=True))
            results["selfcheck"]["d_matrix_vs_frechet_ci"][f"{ds}/{bk}"] = {
                "max_abs_diff": float(np.nanmax(np.abs(D_own - D_ref_matrix))),
                "ok": d_ok,
            }
            if not d_ok:
                raise RuntimeError(f"R4-a 失败: {ds}/{bk} 自算 D 矩阵与 frechet_ci 不一致")

            # cross_subject mean_frechet 对账 T7
            triu = np.triu_indices(len(subjects), k=1)
            mean_d_before = float(D_ref_matrix[triu].mean())
            t7_cs = t7_saved["cross_subject"].get(ds, {}).get(bk, {}).get("mean_frechet")
            if t7_cs is not None:
                # 与 R4-b 同口径的相对容差: mean_riemann 迭代特征分解在跨进程重算时
                # 非位级确定 (SEED_IV/1-4Hz 实测漂移 ~2e-8), 绝对 1e-8 会误报.
                cs_ok = abs(mean_d_before - float(t7_cs)) <= args.ci_rtol * max(1.0, abs(float(t7_cs)))
                results["selfcheck"]["ci_vs_t7_saved"][f"{ds}/{bk}/cross_subject"] = {
                    "recomputed": mean_d_before, "saved": float(t7_cs), "ok": bool(cs_ok)}
                if not cs_ok:
                    raise RuntimeError(f"R4 失败: {ds}/{bk} cross_subject 距离与 T7 存档不一致")

            # 2) ρ̂ (留一源池, 仅用代表点 → label-free)
            rho_dict = rho_leave_one_out(M, subjects)

            # 3) ΔCI (RPA 全体池 fit_transform, T7 口径) + 聚合 CI 对账 T7 (R4-b)
            rows_before = row_mean_offdiag(D_ref_matrix)
            per_aligner_ci = {}
            D_after_rpa = None
            for al_name, AlCls in CI_ALIGNERS.items():
                aligner = AlCls()
                bc = {s: covs[s][band] for s in subjects}
                aligned = aligner.fit_transform(bc, band=band)
                aligned_fmt = {s: {band: aligned[s]} for s in subjects}
                D_a = compute_subject_frechet_matrix(aligned_fmt, subjects, band, MAX_EPOCHS)
                mean_after = float(D_a[triu].mean())
                ci = mean_after / mean_d_before if mean_d_before > 1e-10 else 1.0
                per_aligner_ci[al_name] = ci
                t7_ci = t7_saved["centering_index"].get(ds, {}).get(al_name, {}).get(bk)
                entry = {"recomputed_ci": ci, "saved_ci": (float(t7_ci) if t7_ci is not None else None)}
                if t7_ci is not None:
                    entry["abs_diff"] = abs(ci - float(t7_ci))
                    entry["ok"] = bool(abs(ci - float(t7_ci)) <= args.ci_rtol * max(1.0, abs(float(t7_ci))))
                    if not entry["ok"]:
                        raise RuntimeError(f"R4-b 失败: {ds}/{al_name}/{bk} CI 重算={ci} vs 存档={t7_ci}")
                results["selfcheck"]["ci_vs_t7_saved"][f"{ds}/{al_name}/{bk}"] = entry
                if al_name == "rpa":
                    D_after_rpa = D_a

            rows_after = row_mean_offdiag(D_after_rpa)
            with np.errstate(divide="ignore", invalid="ignore"):
                dci_vals = (rows_after - rows_before) / rows_before

            results["geometry"][ds][bk] = {
                "subjects": list(subjects),
                "rho": [rho_dict[s]["rho"] for s in subjects],
                "d_out": [rho_dict[s]["d_out"] for s in subjects],
                "d_ref": [rho_dict[s]["d_ref"] for s in subjects],
                "dci_rpa": [float(v) for v in dci_vals],
                "mean_d_before_row": [float(v) for v in rows_before],
                "mean_d_after_row_rpa": [float(v) for v in rows_after],
                "dataset_ci_recomputed": per_aligner_ci,
            }

    results["meta"]["elapsed_sec"] = time.time() - t0
    fp = os.path.join(RESULTS_DIR, f"G1_subject_geometry_{ts()}.json")
    with open(fp, "w") as f:
        json.dump(results, f, indent=2)
    log(f"[saved] {fp}")
    log(f"全部自检通过: D矩阵一致性 / T7 CI 对账 / T1 acc 复现 (channels={args.channels})")


if __name__ == "__main__":
    main()
