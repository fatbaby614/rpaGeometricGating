"""
ablation_experiments.py — 消融实验（对应论文 Limitations）
============================================================

4 个消融实验，回应 Reviewer 2 可能提出的问题：

  A: delta-band removal — 重跑 T2/T3 跨数据集迁移，去掉 delta 频段
     （直接验证 "negative transfer 是否由 delta-band 主导"）
  B: Metric comparison — T7 几何量化用 3 种 SPD 度量
     （AIRM vs Log-Euclidean vs Bures-Wasserstein）
  C: Reference mode — T7 CI 用 2 种 reference 模式
     （identity vs grand_mean）
  D: CI threshold sensitivity — 纯分析，基于已有 T7 JSON
     （阈值 0.4/0.5/0.6 下 aligner ranking 稳定性）

用法:
  python experiments/ablation_experiments.py --ablation A
  python experiments/ablation_experiments.py --ablation B
  python experiments/ablation_experiments.py --ablation C
  python experiments/ablation_experiments.py --ablation D
  python experiments/ablation_experiments.py --ablation all

输出:
  results/ablation_A_delta_removal_<timestamp>.json
  results/ablation_B_metric_<timestamp>.json
  results/ablation_C_reference_<timestamp>.json
  results/ablation_D_threshold_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# 添加 code/ 到 path
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE_DIR)

from ric_da_core import (
    BANDS_5, BANDS_8,
    EuclideanAlignment, RiemannianAlignment,
    RiemannianProcrustesAlignment, NoAlignment,
)
from frechet_ci import (
    compute_subject_frechet_matrix, compute_dataset_frechet_distance,
)
from pyriemann.utils.distance import distance
from pyriemann.utils.mean import mean_covariance

# 复用 run_d1_experiments 的数据加载和保存函数
# experiments/ 是 namespace package (无 __init__.py), 需添加到 path
sys.path.insert(0, os.path.join(_CODE_DIR, "experiments"))
from run_d1_experiments import (
    precompute_all_seed, precompute_all_seed_iv, precompute_all_deap,
    list_seed_subs, list_seed_iv_subs, list_deap_subs,
    save, _unify_labels_to_binary, run_cross_dataset,
)

RESULTS_DIR = os.path.join(_CODE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════
# 通用工具
# ════════════════════════════════════════════════════════════════════

def _save_ablation(name: str, results: Dict):
    """保存消融结果到 results/ 目录."""
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = os.path.join(RESULTS_DIR, f"ablation_{name}_{ts}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SAVED] {path}")
    return path


def _load_all_datasets(bands: str = "5band", channels: str = "32ch",
                       n_subjects: int = None) -> Tuple[Dict, Dict]:
    """加载全部 3 个数据集的协方差和标签.
    Returns:
        dataset_covs: {ds_name: {sub: {band: (n_epochs, C, C)}}}
        dataset_y:    {ds_name: {sub: np.ndarray}}
    """
    dataset_covs, dataset_y = {}, {}

    loaders = [
        ("SEED",    list_seed_subs,    precompute_all_seed),
        ("SEED_IV", list_seed_iv_subs, precompute_all_seed_iv),
        ("DEAP",    list_deap_subs,    precompute_all_deap),
    ]
    for ds_name, list_fn, precompute_fn in loaders:
        print(f"\n>>> Loading {ds_name}...")
        subs = list_fn()
        if n_subjects:
            subs = subs[:n_subjects]
        covs, y = precompute_fn(subs, bands=bands, channels=channels)
        if len(covs) > 0:
            dataset_covs[ds_name] = covs
            dataset_y[ds_name] = y

    return dataset_covs, dataset_y


# ════════════════════════════════════════════════════════════════════
# Ablation A: delta-band removal
# ════════════════════════════════════════════════════════════════════

def ablation_a_delta_removal(n_subjects: int = None) -> Dict:
    """消融 A: 去掉 delta 频段，重跑 T2/T3 跨数据集迁移.

    验证: negative transfer 是否由 delta-band 主导。
    若去掉 delta 后 T2/T3 性能显著提升，则 delta-band 是主因；
    若仍然 collapse，则 negative transfer 不是 delta-band-only 问题。

    实现: monkey-patch BANDS_5 为 4-band 版本 (θ/α/β/γ)，
          同时 patch cov_cache.CACHE_VERSION 避免缓存冲突。
    """
    print(f"\n{'#'*60}")
    print(f"# Ablation A: delta-band removal (T2/T3 with θ-α-β-γ)")
    print(f"{'#'*60}")

    # ── monkey-patch BANDS_5 为 4-band (去掉 delta) ──
    BANDS_4_NO_DELTA = [(4, 8), (8, 14), (14, 31), (31, 50)]

    import ric_da_core
    import cov_cache
    from loaders import seed_loader, seed_iv_loader
    import loaders.deap_loader as deap_loader_mod

    original_bands_5_ric = ric_da_core.BANDS_5
    original_bands_5_seed = seed_loader.BANDS_5
    original_bands_5_siv = seed_iv_loader.BANDS_5
    original_bands_5_deap = deap_loader_mod.BANDS_5
    original_cache_ver = cov_cache.CACHE_VERSION

    ric_da_core.BANDS_5 = BANDS_4_NO_DELTA
    seed_loader.BANDS_5 = BANDS_4_NO_DELTA
    seed_iv_loader.BANDS_5 = BANDS_4_NO_DELTA
    # deap_loader 在 import 时从 seed_loader 绑定了 BANDS_5, 需单独 patch
    deap_loader_mod.BANDS_5 = BANDS_4_NO_DELTA
    cov_cache.CACHE_VERSION = "v4_abl_delta"

    print(f"  [PATCH] BANDS_5 = {BANDS_4_NO_DELTA} (delta removed)")
    print(f"  [PATCH] CACHE_VERSION = {cov_cache.CACHE_VERSION}")

    try:
        aligner_names = ["none", "euclidean", "riemann", "rpa"]
        results = {
            "ablation": "A_delta_removal",
            "bands_used": [f"{lo}-{hi}Hz" for lo, hi in BANDS_4_NO_DELTA],
            "bands_removed": ["1-4Hz (delta)"],
            "T2_seed_to_deap": {},
            "T3_deap_to_seed": {},
        }

        # T2: SEED → DEAP
        print(f"\n--- T2: SEED → DEAP (no delta) ---")
        t2 = run_cross_dataset("SEED", "DEAP", aligner_names,
                               channels="32ch", bands="5band",
                               n_subjects=n_subjects)
        for a, r in t2.items():
            results["T2_seed_to_deap"][a] = {
                "acc_mean": r.get("acc_mean"),
                "acc_std": r.get("acc_std"),
                "confusion_matrix": r.get("confusion_matrix"),
            }

        # T3: DEAP → SEED
        print(f"\n--- T3: DEAP → SEED (no delta) ---")
        t3 = run_cross_dataset("DEAP", "SEED", aligner_names,
                               channels="32ch", bands="5band",
                               n_subjects=n_subjects)
        for a, r in t3.items():
            results["T3_deap_to_seed"][a] = {
                "acc_mean": r.get("acc_mean"),
                "acc_std": r.get("acc_std"),
                "confusion_matrix": r.get("confusion_matrix"),
            }

        # 对比基线 (5band, 含 delta) — 从已有 T2/T3 JSON 加载实际值
        def _load_reference_5band(prefix):
            """从 results/ 加载最新的 5band T2/T3 JSON 作为基线."""
            files = sorted([f for f in os.listdir(RESULTS_DIR)
                            if f.startswith(prefix) and f.endswith(".json")
                            and "ablation" not in f])
            if not files:
                return {}
            with open(os.path.join(RESULTS_DIR, files[-1])) as f:
                ref = json.load(f)
            out = {}
            for a in ["none", "euclidean", "riemann", "rpa"]:
                if a in ref and "acc_mean" in ref[a]:
                    out[a] = round(ref[a]["acc_mean"] * 100, 2)
                    # majority baseline: 从 confusion_matrix 推算
                    cm = ref[a].get("confusion_matrix", [])
                    if cm:
                        total = sum(sum(row) for row in cm)
                        if total > 0:
                            majority = max(sum(row) for row in cm) / total * 100
                            out[f"{a}_majority_baseline"] = round(majority, 2)
            return out

        results["reference_5band_T2"] = _load_reference_5band("d1_T2_seed_to_deap")
        results["reference_5band_T3"] = _load_reference_5band("d1_T3_deap_to_seed")
        print(f"\n  [REF] 5band T2 baseline: {results['reference_5band_T2']}")
        print(f"  [REF] 5band T3 baseline: {results['reference_5band_T3']}")

    finally:
        # ── 恢复原始值 ──
        ric_da_core.BANDS_5 = original_bands_5_ric
        seed_loader.BANDS_5 = original_bands_5_seed
        seed_iv_loader.BANDS_5 = original_bands_5_siv
        deap_loader_mod.BANDS_5 = original_bands_5_deap
        cov_cache.CACHE_VERSION = original_cache_ver
        print(f"\n  [RESTORE] BANDS_5 and CACHE_VERSION restored.")

    _save_ablation("A_delta_removal", results)
    return results


# ════════════════════════════════════════════════════════════════════
# Ablation B: Metric comparison (AIRM vs Log-Euclidean vs Bures-Wasserstein)
# ════════════════════════════════════════════════════════════════════

def _frechet_distance_metric(C1, C2, metric="riemann"):
    """支持自定义度量的 Fréchet 距离."""
    return float(distance(C1, C2, metric=metric))


def _compute_subject_frechet_matrix_metric(
    covs, subjects, band, max_epochs=30, metric="riemann"
):
    """计算单频段被试间 Fréchet 距离矩阵 (自定义度量).

    Returns:
        D: (n_valid, n_valid) 对称矩阵
        n_valid: 有效被试数 (mean_covariance 成功且有限的被试)
    """
    # 各被试的均值 (代表点), 用与度量一致的均值函数
    subj_means = {}
    for s in subjects:
        if s not in covs or band not in covs[s]:
            continue
        X = covs[s][band]
        if X is None or len(X) == 0:
            continue
        X = X[:max_epochs] if len(X) > max_epochs else X
        try:
            M = mean_covariance(X, metric=metric)
            if not np.all(np.isfinite(M)):
                continue
            subj_means[s] = M
        except Exception:
            continue

    valid_subs = [s for s in subjects if s in subj_means]
    n_valid = len(valid_subs)
    # 用 NaN 初始化, 避免失败 pair 的 0 值拉低均值
    D = np.full((n_valid, n_valid), np.nan)
    for i, si in enumerate(valid_subs):
        for j, sj in enumerate(valid_subs):
            if i >= j:
                continue
            try:
                d = _frechet_distance_metric(subj_means[si], subj_means[sj],
                                              metric=metric)
                if np.isfinite(d):
                    D[i, j] = d
                    D[j, i] = d
            except Exception:
                continue

    return D, n_valid


def _compute_dataset_frechet_distance_metric(
    covs_a, covs_b, band, max_epochs=30, metric="riemann"
):
    """两个数据集之间的 Fréchet 距离 (自定义度量)."""
    def _dataset_mean(covs):
        all_X = []
        for s in covs:
            if band not in covs[s]:
                continue
            X = covs[s][band]
            if X is None or len(X) == 0:
                continue
            X = X[:max_epochs] if len(X) > max_epochs else X
            all_X.append(X)
        if not all_X:
            return None
        all_X = np.concatenate(all_X, axis=0)
        try:
            return mean_covariance(all_X, metric=metric)
        except Exception:
            return None

    Ma = _dataset_mean(covs_a)
    Mb = _dataset_mean(covs_b)
    if Ma is None or Mb is None:
        return float("nan")
    return _frechet_distance_metric(Ma, Mb, metric=metric)


def _full_frechet_analysis_metric(
    dataset_covs, dataset_names, bands, max_epochs=30,
    aligner_names=None, metric="riemann", verbose=True,
):
    """完整的 Fréchet 分析 (自定义度量).

    对齐器固定为 AIRM-based (RA/RPA), 但用不同度量"测量"对齐效果。
    即: aligner 不变, CI 的距离计算用指定 metric。
    """
    if aligner_names is None:
        aligner_names = ["euclidean", "riemann", "rpa"]

    ALIGNER_CLASSES = {
        "none": NoAlignment,
        "euclidean": EuclideanAlignment,
        "riemann": RiemannianAlignment,
        "rpa": RiemannianProcrustesAlignment,
    }

    t0 = time.time()
    results = {"cross_subject": {}, "cross_dataset": {}, "centering_index": {}}

    for band in bands:
        band_key = f"{band[0]}-{band[1]}Hz"
        if verbose:
            print(f"\n--- [{metric}] Band {band_key} ---")

        for ds_name in dataset_names:
            if ds_name not in dataset_covs:
                continue
            subs = list(dataset_covs[ds_name].keys())
            if len(subs) < 2:
                continue

            D_before, n_valid = _compute_subject_frechet_matrix_metric(
                dataset_covs[ds_name], subs, band, max_epochs, metric=metric
            )
            if n_valid < 2:
                continue
            triu_idx = np.triu_indices(n_valid, k=1)
            triu_vals = D_before[triu_idx]
            if np.all(np.isnan(triu_vals)):
                # 所有 pair 都失败, 无法计算均值
                mean_d_before = float("nan")
            else:
                mean_d_before = float(np.nanmean(triu_vals))

            results["cross_subject"].setdefault(ds_name, {})[band_key] = {
                "mean_frechet": mean_d_before, "n_subjects": n_valid,
            }
            if verbose:
                print(f"  {ds_name} cross-subject: {mean_d_before:.4f} ({n_valid} subs)")

            for aligner_name in aligner_names:
                if aligner_name == "none":
                    ci = 1.0
                else:
                    aligner = ALIGNER_CLASSES[aligner_name]()
                    band_covs = {s: dataset_covs[ds_name][s][band] for s in subs
                                 if s in dataset_covs[ds_name] and band in dataset_covs[ds_name][s]}
                    if not band_covs:
                        continue
                    try:
                        aligned = aligner.fit_transform(band_covs, band=band)
                    except Exception as e:
                        print(f"  [WARN] align failed: {e}")
                        ci = float("nan")
                        results["centering_index"].setdefault(
                            ds_name, {}).setdefault(aligner_name, {})[band_key] = ci
                        continue

                    aligned_has_nan = any(not np.all(np.isfinite(aligned[s]))
                                          for s in aligned)
                    if aligned_has_nan:
                        ci = float("nan")
                    else:
                        aligned_fmt = {s: {band: aligned[s]} for s in aligned}
                        D_after, n_after = _compute_subject_frechet_matrix_metric(
                            aligned_fmt, list(aligned.keys()), band, max_epochs, metric=metric
                        )
                        if n_after < 2:
                            # 少于 2 个有效被试, 无法计算 pairwise 距离
                            ci = float("nan")
                        else:
                            triu_after = np.triu_indices(n_after, k=1)
                            triu_after_vals = D_after[triu_after]
                            if np.all(np.isnan(triu_after_vals)):
                                mean_d_after = float("nan")
                            else:
                                mean_d_after = float(np.nanmean(triu_after_vals))

                            if not np.isfinite(mean_d_before):
                                ci = float("nan")
                            elif mean_d_before < 1e-10:
                                ci = 1.0
                            elif not np.isfinite(mean_d_after):
                                ci = float("nan")
                            else:
                                ci = float(mean_d_after / mean_d_before)

                results["centering_index"].setdefault(
                    ds_name, {}).setdefault(aligner_name, {})[band_key] = ci
                if verbose:
                    print(f"    CI[{ds_name}/{aligner_name}]: {ci:.4f}")

        # 跨数据集距离
        for i, ds_a in enumerate(dataset_names):
            for ds_b in dataset_names[i+1:]:
                if ds_a not in dataset_covs or ds_b not in dataset_covs:
                    continue
                d = _compute_dataset_frechet_distance_metric(
                    dataset_covs[ds_a], dataset_covs[ds_b], band, max_epochs, metric=metric
                )
                results["cross_dataset"].setdefault(
                    f"{ds_a}_vs_{ds_b}", {})[band_key] = float(d)
                if verbose:
                    print(f"  {ds_a} vs {ds_b}: {d:.4f}")

    results["elapsed_sec"] = float(time.time() - t0)
    results["metric"] = metric
    return results


def ablation_b_metric_comparison(n_subjects: int = None) -> Dict:
    """消融 B: 对比 3 种 SPD 度量 (AIRM / Log-Euclidean / Bures-Wasserstein).

    对齐器固定为 AIRM-based RA/RPA (与论文一致),
    但用不同度量"测量" cross-subject / cross-dataset 距离和 CI。
    验证: 几何判据 (ρ, CI) 是否对度量选择稳健。
    """
    print(f"\n{'#'*60}")
    print(f"# Ablation B: Metric comparison (AIRM vs LogE vs BW)")
    print(f"{'#'*60}")

    # 加载数据 (一次加载, 三次分析)
    dataset_covs, _ = _load_all_datasets(bands="5band", channels="32ch",
                                          n_subjects=n_subjects)
    if len(dataset_covs) < 2:
        print("  [SKIP] 数据不足")
        return {}

    results = {
        "ablation": "B_metric_comparison",
        "metrics": ["riemann", "logeuclid", "wasserstein"],
        "per_metric": {},
    }

    for metric in ["riemann", "logeuclid", "wasserstein"]:
        print(f"\n{'='*40}")
        print(f"= Metric: {metric}")
        print(f"{'='*40}")
        res = _full_frechet_analysis_metric(
            dataset_covs, list(dataset_covs.keys()), BANDS_5,
            aligner_names=["euclidean", "riemann", "rpa"],
            metric=metric, verbose=True,
        )
        results["per_metric"][metric] = res

    # 汇总: ρ = d_ds / d_subj (delta band) 对比
    print(f"\n{'='*40}")
    print(f"= Summary: ρ_delta = d_ds_delta / d_subj_delta")
    print(f"{'='*40}")
    rho_summary = {}
    for metric in ["riemann", "logeuclid", "wasserstein"]:
        res = results["per_metric"][metric]
        # SEED vs DEAP, delta band
        d_ds = res["cross_dataset"].get("SEED_vs_DEAP", {}).get("1-4Hz")
        d_subj_seed = res["cross_subject"].get("SEED", {}).get("1-4Hz", {}).get("mean_frechet")
        if d_ds and d_subj_seed and d_subj_seed > 0:
            rho = d_ds / d_subj_seed
            rho_summary[metric] = {"d_ds_delta": d_ds, "d_subj_seed_delta": d_subj_seed,
                                    "rho_delta": rho}
            print(f"  {metric}: ρ_delta = {rho:.3f} (d_ds={d_ds:.2f}, d_subj={d_subj_seed:.2f})")
    results["rho_delta_summary"] = rho_summary

    _save_ablation("B_metric", results)
    return results


# ════════════════════════════════════════════════════════════════════
# Ablation C: Reference mode (identity vs grand_mean)
# ════════════════════════════════════════════════════════════════════

def _full_frechet_analysis_reference(
    dataset_covs, dataset_names, bands, max_epochs=30,
    aligner_names=None, reference="identity", verbose=True,
):
    """完整 Fréchet 分析 (自定义 reference 模式).

    用 AIRM 度量, 但 aligner 的 reference 可选 identity 或 grand_mean。
    验证: CI 是否对 reference 选择稳健。
    """
    if aligner_names is None:
        aligner_names = ["euclidean", "riemann", "rpa"]

    ALIGNER_CLASSES = {
        "euclidean": EuclideanAlignment,
        "riemann": RiemannianAlignment,
        "rpa": RiemannianProcrustesAlignment,
    }

    t0 = time.time()
    results = {"centering_index": {}, "reference": reference}

    for band in bands:
        band_key = f"{band[0]}-{band[1]}Hz"
        if verbose:
            print(f"\n--- [ref={reference}] Band {band_key} ---")

        for ds_name in dataset_names:
            if ds_name not in dataset_covs:
                continue
            subs = list(dataset_covs[ds_name].keys())
            if len(subs) < 2:
                continue

            # 对齐前距离 (不依赖 reference)
            # 注意: compute_subject_frechet_matrix 内部 mean_riemann 无 try-except,
            # 对齐后协方差若奇异会抛 LinAlgError, 需显式保护
            try:
                D_before = compute_subject_frechet_matrix(
                    dataset_covs[ds_name], subs, band, max_epochs
                )
            except Exception as e:
                print(f"  [WARN] compute_subject_frechet_matrix (before) failed: {e}")
                for aligner_name in aligner_names:
                    results["centering_index"].setdefault(
                        ds_name, {}).setdefault(aligner_name, {})[band_key] = float("nan")
                continue
            n = len(subs)
            triu_idx = np.triu_indices(n, k=1)
            mean_d_before = float(D_before[triu_idx].mean())

            for aligner_name in aligner_names:
                # 用指定 reference 实例化 aligner
                aligner = ALIGNER_CLASSES[aligner_name](reference=reference)
                band_covs = {s: dataset_covs[ds_name][s][band] for s in subs
                             if s in dataset_covs[ds_name] and band in dataset_covs[ds_name][s]}
                if not band_covs:
                    continue
                try:
                    aligned = aligner.fit_transform(band_covs, band=band)
                except Exception as e:
                    print(f"  [WARN] align failed: {e}")
                    ci = float("nan")
                    results["centering_index"].setdefault(
                        ds_name, {}).setdefault(aligner_name, {})[band_key] = ci
                    continue

                aligned_has_nan = any(not np.all(np.isfinite(aligned[s]))
                                      for s in aligned)
                if aligned_has_nan:
                    ci = float("nan")
                else:
                    aligned_fmt = {s: {band: aligned[s]} for s in aligned}
                    try:
                        D_after = compute_subject_frechet_matrix(
                            aligned_fmt, list(aligned.keys()), band, max_epochs
                        )
                    except Exception as e:
                        print(f"  [WARN] compute_subject_frechet_matrix (after) failed: {e}")
                        ci = float("nan")
                        results["centering_index"].setdefault(
                            ds_name, {}).setdefault(aligner_name, {})[band_key] = ci
                        continue
                    n_after = len(aligned)
                    if n_after < 2:
                        # 少于 2 个有效被试, 无法计算 pairwise 距离
                        ci = float("nan")
                    else:
                        triu_after = np.triu_indices(n_after, k=1)
                        mean_d_after = float(D_after[triu_after].mean())

                        if not np.isfinite(mean_d_before):
                            ci = float("nan")
                        elif mean_d_before < 1e-10:
                            ci = 1.0
                        elif not np.isfinite(mean_d_after):
                            ci = float("nan")
                        else:
                            ci = float(mean_d_after / mean_d_before)

                results["centering_index"].setdefault(
                    ds_name, {}).setdefault(aligner_name, {})[band_key] = ci
                if verbose:
                    print(f"    CI[{ds_name}/{aligner_name}/{reference}]: {ci:.4f}")

    results["elapsed_sec"] = float(time.time() - t0)
    return results


def ablation_c_reference_mode(n_subjects: int = None) -> Dict:
    """消融 C: 对比 2 种 reference 模式 (identity vs grand_mean).

    验证: CI 的 aligner ranking 是否对 reference 选择稳健。
    """
    print(f"\n{'#'*60}")
    print(f"# Ablation C: Reference mode (identity vs grand_mean)")
    print(f"{'#'*60}")

    dataset_covs, _ = _load_all_datasets(bands="5band", channels="32ch",
                                          n_subjects=n_subjects)
    if len(dataset_covs) < 2:
        print("  [SKIP] 数据不足")
        return {}

    results = {
        "ablation": "C_reference_mode",
        "references": ["identity", "grand_mean"],
        "per_reference": {},
    }

    for ref in ["identity", "grand_mean"]:
        print(f"\n{'='*40}")
        print(f"= Reference: {ref}")
        print(f"{'='*40}")
        res = _full_frechet_analysis_reference(
            dataset_covs, list(dataset_covs.keys()), BANDS_5,
            aligner_names=["euclidean", "riemann", "rpa"],
            reference=ref, verbose=True,
        )
        results["per_reference"][ref] = res

    # 汇总: 比较 2 种 reference 下的 CI ranking
    print(f"\n{'='*40}")
    print(f"= CI ranking comparison (RPA < 0.5 count)")
    print(f"{'='*40}")
    for ref in ["identity", "grand_mean"]:
        ci_data = results["per_reference"][ref]["centering_index"]
        rpa_below_05 = 0
        rpa_total = 0
        for ds in ci_data:
            if "rpa" in ci_data[ds]:
                for band, ci in ci_data[ds]["rpa"].items():
                    rpa_total += 1
                    if np.isfinite(ci) and ci < 0.5:
                        rpa_below_05 += 1
        print(f"  {ref}: RPA CI<0.5 in {rpa_below_05}/{rpa_total} entries")
        results[f"rpa_ci_below_05_{ref}"] = f"{rpa_below_05}/{rpa_total}"

    _save_ablation("C_reference", results)
    return results


# ════════════════════════════════════════════════════════════════════
# Ablation D: CI threshold sensitivity (纯分析)
# ════════════════════════════════════════════════════════════════════

def ablation_d_threshold_sensitivity(t7_json_path: str = None) -> Dict:
    """消融 D: CI 阈值敏感性分析 (纯分析, 不需要重跑实验).

    基于 T7 已有结果, 分析阈值 0.4/0.5/0.6 下:
      1. RPA 通过 CI<threshold 的 entry 数量
      2. aligner ranking (RPA vs RA vs EA) 的稳定性
    """
    print(f"\n{'#'*60}")
    print(f"# Ablation D: CI threshold sensitivity (analysis only)")
    print(f"{'#'*60}")

    # 加载已有 T7 结果
    if t7_json_path is None:
        # 自动查找最新的 T7 JSON
        t7_files = sorted([f for f in os.listdir(RESULTS_DIR)
                           if f.startswith("d1_T7_") and f.endswith(".json")])
        if not t7_files:
            print("  [ERROR] 未找到 T7 结果文件")
            return {}
        t7_json_path = os.path.join(RESULTS_DIR, t7_files[-1])
    print(f"  Loading: {t7_json_path}")

    with open(t7_json_path) as f:
        t7 = json.load(f)

    ci_data = t7["centering_index"]
    thresholds = [0.4, 0.5, 0.6]

    results = {
        "ablation": "D_threshold_sensitivity",
        "source_file": os.path.basename(t7_json_path),
        "thresholds": thresholds,
        "per_threshold": {},
    }

    for thr in thresholds:
        print(f"\n--- Threshold CI < {thr} ---")
        thr_result = {"datasets": {}}
        for ds in ci_data:
            ds_result = {}
            for aligner in ["euclidean", "riemann", "rpa"]:
                if aligner not in ci_data[ds]:
                    continue
                ci_vals = list(ci_data[ds][aligner].values())
                ci_finite = [v for v in ci_vals if np.isfinite(v)]
                n_below = sum(1 for v in ci_finite if v < thr)
                n_total = len(ci_finite)
                ds_result[aligner] = {
                    "n_below_threshold": n_below,
                    "n_total": n_total,
                    "fraction_below": n_below / n_total if n_total > 0 else 0,
                    "ci_values": ci_data[ds][aligner],
                }
                print(f"  {ds}/{aligner}: {n_below}/{n_total} below {thr} "
                      f"(frac={n_below/n_total:.2f})")
            thr_result["datasets"][ds] = ds_result
        results["per_threshold"][str(thr)] = thr_result

    # ── ranking 稳定性: 对每个 (dataset, band), 比较 3 个 aligner 的 CI 排名 ──
    print(f"\n--- Aligner ranking stability across thresholds ---")
    ranking_stability = {}
    for ds in ci_data:
        if "rpa" not in ci_data[ds] or "riemann" not in ci_data[ds]:
            continue
        for band in ci_data[ds]["rpa"]:
            if band not in ci_data[ds]["riemann"] or band not in ci_data[ds].get("euclidean", {}):
                continue
            ci_rpa = ci_data[ds]["rpa"][band]
            ci_ra = ci_data[ds]["riemann"][band]
            ci_ea = ci_data[ds]["euclidean"][band]
            if not all(np.isfinite(v) for v in [ci_rpa, ci_ra, ci_ea]):
                continue
            # ranking: 1=best (lowest CI)
            vals = {"rpa": ci_rpa, "riemann": ci_ra, "euclidean": ci_ea}
            ranked = sorted(vals, key=lambda k: vals[k])
            ranking_stability[f"{ds}/{band}"] = {
                "ranking": ranked,
                "ci_values": vals,
            }

    # 统计每种 ranking pattern 出现次数
    from collections import Counter
    patterns = Counter()
    for v in ranking_stability.values():
        patterns[tuple(v["ranking"])] += 1
    print(f"\n  Ranking pattern frequency (n={sum(patterns.values())}):")
    for pattern, count in patterns.most_common():
        print(f"    {' < '.join(pattern)}: {count}")
    results["ranking_stability"] = {
        "pattern_frequency": {">".join(k): v for k, v in patterns.items()},
        "total_entries": sum(patterns.values()),
        "details": ranking_stability,
    }

    # ── 关键结论: 阈值变化是否改变 Q1 结论 (RPA CI<0.5 across all 15 entries) ──
    print(f"\n--- Q1 conclusion stability ---")
    for thr in thresholds:
        rpa_all_below = True
        n_total = 0
        for ds in ci_data:
            if "rpa" not in ci_data[ds]:
                continue
            for band, ci in ci_data[ds]["rpa"].items():
                n_total += 1
                if not np.isfinite(ci) or ci >= thr:
                    rpa_all_below = False
        print(f"  CI<{thr}: RPA all below = {rpa_all_below} ({n_total} entries)")
        results[f"q1_rpa_all_below_{thr}"] = rpa_all_below

    _save_ablation("D_threshold", results)
    return results


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ablation experiments")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["A", "B", "C", "D", "all"],
                        help="Which ablation to run (default: all)")
    parser.add_argument("--n_subjects", type=int, default=None,
                        help="Limit subjects per dataset (debug)")
    parser.add_argument("--t7_json", type=str, default=None,
                        help="Path to T7 JSON for ablation D")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"= Ablation Experiments")
    print(f"= Ablation: {args.ablation}")
    print(f"{'='*60}")

    if args.ablation in ["A", "all"]:
        ablation_a_delta_removal(n_subjects=args.n_subjects)
    if args.ablation in ["B", "all"]:
        ablation_b_metric_comparison(n_subjects=args.n_subjects)
    if args.ablation in ["C", "all"]:
        ablation_c_reference_mode(n_subjects=args.n_subjects)
    if args.ablation in ["D", "all"]:
        ablation_d_threshold_sensitivity(t7_json_path=args.t7_json)

    print(f"\n{'='*60}")
    print(f"= All ablations done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
