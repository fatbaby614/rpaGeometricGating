"""
frechet_ci.py — Fréchet 距离 + Centering Index 量化分析
=========================================================

复用自 RiemannianDomainAdaptation/code/frechet_analysis.py，扩展支持跨数据集量化。

功能:
    1. 跨被试 Fréchet 距离矩阵 (单数据集内)
    2. 跨数据集 Fréchet 距离 (SEED vs DEAP vs SEED-IV)
    3. Centering Index (CI) = 对齐后距离 / 对齐前距离

度量:
    Fréchet distance (AIRM) δ_F(C1, C2) = ||logm(C1^{-1/2} · C2 · C1^{-1/2})||_F
    CI < 1: 对齐有效
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
from pyriemann.utils.distance import distance
from pyriemann.utils.mean import mean_riemann


def frechet_distance(C1: np.ndarray, C2: np.ndarray) -> float:
    """AIRM (Affine-Invariant Riemannian Metric) 距离."""
    return float(distance(C1, C2, metric="riemann"))


def compute_subject_frechet_matrix(
    covs: Dict[str, Dict[Tuple[float, float], np.ndarray]],
    subjects: List[str],
    band: Tuple[float, float],
    max_epochs: int = 30,
) -> np.ndarray:
    """计算被试间 Fréchet 距离矩阵 (单频段).

    优化: 每个被试的黎曼均值只算一次, 缓存后复用.
    原始实现: O(n^2) 对, 每对算 2 次 mean_riemann = 2*n*(n-1) 次
    优化后:    O(n) 次 mean_riemann + O(n^2) 次 frechet_distance

    Args:
        covs: {sub: {band: (n_epochs, C, C)}}
        subjects: 被试 ID 列表
        band: 频段 (lo, hi)
        max_epochs: 每被试最多用前 N 个 epoch

    Returns:
        D: (n_subjects, n_subjects) 对称矩阵
    """
    # 预计算每个被试的黎曼均值 (只算一次, 复用)
    subj_means = {}
    for s in subjects:
        covs_s = covs[s][band][:max_epochs]
        # 空输入保护: covs_s 为 (0, C, C) 时 mean_riemann 返回 NaN 矩阵 (带 RuntimeWarning),
        # NaN 会通过 frechet_distance 传播到整行整列 D, 最终污染 mean_d_before/mean_d_after,
        # 触发 full_frechet_analysis L224 的 NaN > 1e-10 == False 误判 CI=1.0.
        # 实际场景下 precompute_all_* 会跳过空被试, 但 4ch/8ch + epoch < n_ch 时
        # 协方差可能奇异, mean_riemann 不收敛返回 NaN, 必须显式检查.
        if covs_s.shape[0] == 0:
            print(f"  [frechet][WARN] 被试 {s} band={band} covs 为空, "
                  f"mean_riemann 会返回 NaN, D 矩阵将含 NaN", flush=True)
        riem_mean = mean_riemann(covs_s, maxiter=30)
        # 对称化: mean_riemann 迭代可能引入微小数值不对称
        riem_mean = (riem_mean + riem_mean.T) / 2
        if not np.all(np.isfinite(riem_mean)):
            print(f"  [frechet][WARN] 被试 {s} band={band} mean_riemann 含 NaN/Inf "
                  f"(可能因协方差奇异或迭代不收敛), D 矩阵将含 NaN", flush=True)
        subj_means[s] = riem_mean

    n = len(subjects)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            # 用预计算的黎曼均值算距离
            # 注: 若 subj_means 含 NaN, frechet_distance 会返回 NaN,
            # NaN 写入 D 后会传播到 mean_d_before/after, 上层需检查.
            try:
                d = frechet_distance(subj_means[subjects[i]], subj_means[subjects[j]])
            except (ValueError, np.linalg.LinAlgError) as e:
                # 奇异矩阵: invsqrtm 对零特征值产生 inf, distance 会抛 LinAlgError
                # 或返回 inf/NaN. 标记为 NaN 让上层检测.
                print(f"  [frechet][WARN] {subjects[i]} vs {subjects[j]} "
                      f"frechet_distance 失败: {e}, 标记为 NaN", flush=True)
                d = float('nan')
            D[i, j] = d
            D[j, i] = d
    return D


def compute_dataset_frechet_distance(
    covs_a: Dict[str, Dict],
    covs_b: Dict[str, Dict],
    band: Tuple[float, float],
    max_epochs: int = 30,
) -> float:
    """计算两个数据集之间的 Fréchet 距离.
    
    用每个数据集所有被试的 SPD 黎曼均值代表该数据集。
    
    Returns:
        d: 标量距离
    """
    # 空数据集保护: covs_a/covs_b 为空 dict 时 np.concatenate([]) 抛 ValueError,
    # 或所有被试 epoch=0 时 all_covs 为 (0, C, C), mean_riemann 返回 NaN.
    if len(covs_a) == 0 or len(covs_b) == 0:
        print(f"  [frechet][WARN] compute_dataset_frechet_distance: "
              f"空数据集 (a={len(covs_a)}, b={len(covs_b)}), 返回 NaN", flush=True)
        return float('nan')

    # 数据集 A 的黎曼均值
    all_covs_a = np.concatenate(
        [covs_a[s][band][:max_epochs] for s in covs_a], axis=0
    )
    if all_covs_a.shape[0] == 0:
        print(f"  [frechet][WARN] 数据集 A 所有被试 epoch=0, 返回 NaN", flush=True)
        return float('nan')
    mean_a = mean_riemann(all_covs_a, maxiter=50)
    mean_a = (mean_a + mean_a.T) / 2  # 对称化

    # 数据集 B 的黎曼均值
    all_covs_b = np.concatenate(
        [covs_b[s][band][:max_epochs] for s in covs_b], axis=0
    )
    if all_covs_b.shape[0] == 0:
        print(f"  [frechet][WARN] 数据集 B 所有被试 epoch=0, 返回 NaN", flush=True)
        return float('nan')
    mean_b = mean_riemann(all_covs_b, maxiter=50)
    mean_b = (mean_b + mean_b.T) / 2  # 对称化

    # NaN 检查: mean_riemann 不收敛时返回 NaN, frechet_distance 会传播 NaN
    if not np.all(np.isfinite(mean_a)) or not np.all(np.isfinite(mean_b)):
        print(f"  [frechet][WARN] 数据集均值含 NaN/Inf, 返回 NaN", flush=True)
        return float('nan')

    try:
        return frechet_distance(mean_a, mean_b)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"  [frechet][WARN] frechet_distance 失败: {e}, 返回 NaN", flush=True)
        return float('nan')


def compute_centering_index(
    covs_before: Dict[str, Dict],
    covs_after: Dict[str, Dict],
    subjects: List[str],
    band: Tuple[float, float],
    max_epochs: int = 30,
) -> float:
    """计算 Centering Index (CI).
    
    CI = mean(δ_F(after)) / mean(δ_F(before))
    CI < 1: 对齐有效
    
    Args:
        covs_before: 对齐前的协方差
        covs_after: 对齐后的协方差
    
    Returns:
        ci: Centering Index
    """
    D_before = compute_subject_frechet_matrix(covs_before, subjects, band, max_epochs)
    D_after = compute_subject_frechet_matrix(covs_after, subjects, band, max_epochs)
    
    # 取上三角 (不含对角线) 的均值
    n = len(subjects)
    triu_idx = np.triu_indices(n, k=1)
    
    mean_before = D_before[triu_idx].mean()
    mean_after = D_after[triu_idx].mean()

    # NaN/Inf 防护 + 防除零:
    # - 若 mean_before 为 NaN (因 mean_riemann 不收敛或协方差奇异),
    #   NaN < 1e-10 在 Python 中返回 False, 会走到 return NaN/NaN = NaN,
    #   虽然传播了 NaN 但没显式标记. 改为显式返回 NaN 并标记.
    # - 若 mean_before < 1e-10 (对齐前距离已接近 0, 数据集本就同质),
    #   CI 无意义, 返回 1.0 (对齐无效).
    # - 若 mean_after 为 NaN/Inf (对齐失败), 返回 NaN 标记问题.
    if not np.isfinite(mean_before):
        print(f"  [CI][WARN] mean_before 含 NaN/Inf, CI 无意义, 返回 NaN",
              flush=True)
        return float('nan')
    if mean_before < 1e-10:
        return 1.0
    if not np.isfinite(mean_after):
        print(f"  [CI][WARN] mean_after 含 NaN/Inf (对齐可能失败), 返回 NaN",
              flush=True)
        return float('nan')

    return float(mean_after / mean_before)


def full_frechet_analysis(
    dataset_covs: Dict[str, Dict[str, Dict]],
    dataset_names: List[str],
    bands: List[Tuple[float, float]],
    max_epochs: int = 30,
    aligner_names: List[str] = None,
    verbose: bool = True,
) -> Dict:
    """完整的 Fréchet 分析: 跨被试 + 跨数据集 + CI.

    Args:
        dataset_covs: {dataset_name: {sub: {band: (n_epochs, C, C)}}}
        dataset_names: 数据集名称列表
        bands: 频段列表
        max_epochs: 每被试最多用前 N 个 epoch
        aligner_names: 要计算 CI 的对齐器列表 (默认 ['euclidean','riemann','rpa'])
            CI = mean(δ_F 对齐后) / mean(δ_F 对齐前)
            CI < 1: 对齐有效
            CI < 0.5: 差异压缩超过 50% (验证 H1)
            none 对齐器 CI 恒为 1.0, 不计算

    Returns:
        dict: 完整分析结果
            cross_subject: {dataset: {band: {mean_frechet, n_subjects}}}
            cross_dataset: {dsA_vs_dsB: {band: distance}}
            centering_index: {dataset: {aligner: {band: ci}}}
    """
    from ric_da_core import (EuclideanAlignment, RiemannianAlignment,
                              RiemannianProcrustesAlignment, NoAlignment)

    if aligner_names is None:
        # 默认包含 RPA, 用于验证 H1
        aligner_names = ["euclidean", "riemann", "rpa"]

    # 对齐器实例映射 (新增 rpa)
    ALIGNER_CLASSES = {
        "none": NoAlignment,
        "euclidean": EuclideanAlignment,
        "riemann": RiemannianAlignment,
        "rpa": RiemannianProcrustesAlignment,
    }

    t0 = time.time()
    results = {
        'cross_subject': {},          # 跨被试距离 (对齐前)
        'cross_dataset': {},          # 跨数据集距离
        'centering_index': {},        # CI: {dataset: {aligner: {band: ci}}}
    }

    for band in bands:
        band_key = f"{band[0]}-{band[1]}Hz"
        if verbose:
            print(f"\n--- Band {band_key} ---")

        # 跨被试距离 (每个数据集内, 对齐前) + CI
        for ds_name in dataset_names:
            if ds_name not in dataset_covs:
                continue
            subs = list(dataset_covs[ds_name].keys())
            if len(subs) < 2:
                continue

            # 对齐前的 Fréchet 距离矩阵
            D_before = compute_subject_frechet_matrix(
                dataset_covs[ds_name], subs, band, max_epochs
            )
            n = len(subs)
            triu_idx = np.triu_indices(n, k=1)
            mean_d_before = float(D_before[triu_idx].mean())

            results['cross_subject'].setdefault(ds_name, {})[band_key] = {
                'mean_frechet': mean_d_before,
                'n_subjects': n,
            }
            if verbose:
                print(f"  {ds_name} cross-subject: {mean_d_before:.4f} ({n} subs)")

            # 对各 aligner 计算 CI
            for aligner_name in aligner_names:
                if aligner_name == "none":
                    ci = 1.0  # 无对齐, CI 恒为 1
                else:
                    aligner = ALIGNER_CLASSES[aligner_name]()
                    # 对齐: {sub: (n_epochs, C, C)}
                    band_covs = {s: dataset_covs[ds_name][s][band] for s in subs}
                    aligned = aligner.fit_transform(band_covs, band=band)
                    # NaN 检查: aligner.fit_transform 后若 aligned 含 NaN
                    # (因 mean_riemann 不收敛或协方差奇异), 会传播到 D_after
                    # 和 mean_d_after, 最终污染 CI. 必须显式检查.
                    aligned_has_nan = any(not np.all(np.isfinite(aligned[s]))
                                          for s in subs)
                    if aligned_has_nan:
                        print(f"  [CI][WARN] {ds_name}/{aligner_name}/band={band_key}: "
                              f"aligned 含 NaN/Inf (对齐失败), CI=NaN", flush=True)
                        ci = float('nan')
                    else:
                        # 转回 {sub: {band: (n_epochs, C, C)}} 格式
                        aligned_fmt = {s: {band: aligned[s]} for s in subs}
                        D_after = compute_subject_frechet_matrix(
                            aligned_fmt, subs, band, max_epochs
                        )
                        mean_d_after = float(D_after[triu_idx].mean())
                        # NaN/Inf 防护 + 防除零 (与 compute_centering_index 一致):
                        # - mean_d_before 为 NaN: 数据集本身有问题, CI=NaN
                        # - mean_d_before < 1e-10: 对齐前已同质, CI=1.0
                        # - mean_d_after 为 NaN: 对齐后有问题, CI=NaN
                        # ⚠️ 旧代码 `mean_d_before > 1e-10` 在 NaN 时返回 False,
                        # 会走 else 分支 ci=1.0, 把"数据有问题"误报为"对齐无效",
                        # 这是论文级 bug (静默得出错误结论).
                        if not np.isfinite(mean_d_before):
                            print(f"  [CI][WARN] {ds_name}/{aligner_name}/band={band_key}: "
                                  f"mean_d_before 含 NaN/Inf, CI=NaN", flush=True)
                            ci = float('nan')
                        elif mean_d_before < 1e-10:
                            ci = 1.0
                        elif not np.isfinite(mean_d_after):
                            print(f"  [CI][WARN] {ds_name}/{aligner_name}/band={band_key}: "
                                  f"mean_d_after 含 NaN/Inf, CI=NaN", flush=True)
                            ci = float('nan')
                        else:
                            ci = float(mean_d_after / mean_d_before)

                results['centering_index'].setdefault(ds_name, {}).setdefault(aligner_name, {})[band_key] = ci
                if verbose:
                    print(f"    CI[{ds_name}/{aligner_name}]: {ci:.4f}")

        # 跨数据集距离
        for i, ds_a in enumerate(dataset_names):
            for ds_b in dataset_names[i+1:]:
                if ds_a not in dataset_covs or ds_b not in dataset_covs:
                    continue
                d = compute_dataset_frechet_distance(
                    dataset_covs[ds_a], dataset_covs[ds_b], band, max_epochs
                )
                results['cross_dataset'].setdefault(f"{ds_a}_vs_{ds_b}", {})[band_key] = float(d)
                if verbose:
                    print(f"  {ds_a} vs {ds_b}: {d:.4f}")

    results['elapsed_sec'] = float(time.time() - t0)
    return results
