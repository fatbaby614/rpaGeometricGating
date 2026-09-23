"""
experiments/run_d1_experiments.py — D1 主实验
==============================================

执行 D1 研究方向的实验矩阵:
    T1: 单数据集 LOSO baseline (SEED / SEED-IV / DEAP)
    T2: SEED→DEAP 跨库迁移
    T3: DEAP→SEED 跨库迁移
    T4: SEED↔SEED-IV 同源跨标签
    T5: 通道敏感性 (32/16/8/4ch)
    T7: Fréchet + CI 量化分析

用法:
    python experiments/run_d1_experiments.py --tables T1,T2,T3,T4,T7
    python experiments/run_d1_experiments.py --tables T1 --datasets SEED --channels 32ch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

# 路径修正: 让本文件能从 code/ 目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import SEED_ROOT, SEED_IV_ROOT, DEAP_ROOT
print(f"[paths] SEED:    {SEED_ROOT}")
print(f"[paths] SEED_IV: {SEED_IV_ROOT}")
print(f"[paths] DEAP:    {DEAP_ROOT}")

from ric_da_core import (
    NoAlignment, EuclideanAlignment, RiemannianAlignment,
    RiemannianProcrustesAlignment,
    evaluate_loso_classification, evaluate_cross_dataset,
    paired_t_test, permutation_test, compare_aligners, compare_all_datasets,
    BANDS_5, BANDS_8
)
from frechet_ci import full_frechet_analysis
from loaders.seed_loader import precompute_all_seed, list_subjects as list_seed_subs
from loaders.seed_iv_loader import precompute_all_seed_iv, list_subjects as list_seed_iv_subs
from loaders.deap_loader import precompute_all_deap, list_subjects as list_deap_subs

# 输出目录: 项目根/results/ (本文件在 code/experiments/, 上三级到项目根)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"[OUTPUT] 结果将保存到: {RESULTS_DIR}")


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str):
    """带时间戳的日志打印 (flush 确保实时写入日志文件)."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def save(name: str, data: Dict) -> str:
    fp = os.path.join(RESULTS_DIR, f"d1_{name}_{ts()}.json")
    with open(fp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [saved] {fp}")
    return fp


# 对齐器
ALIGNERS = {
    "none":      NoAlignment,
    "euclidean": EuclideanAlignment,
    "riemann":   RiemannianAlignment,
    "rpa":       RiemannianProcrustesAlignment,
}


# ════════════════════════════════════════════════════════════════════
# T1: 单数据集 LOSO baseline
# ════════════════════════════════════════════════════════════════════

def run_T1(datasets: List[str], aligner_names: List[str],
           channels: str = "32ch", bands: str = "5band",
           n_subjects: int = None,
           save_result: bool = True) -> Dict:
    """T1: 单数据集 LOSO baseline.

    Args:
        save_result: 是否保存到 results/ (T5 调用时设 False 避免覆盖 T1)
    """
    print(f"\n{'#'*60}")
    print(f"# T1: Single-Dataset LOSO Baseline")
    print(f"# Datasets: {datasets}, Aligners: {aligner_names}")
    print(f"# Channels: {channels}, Bands: {bands}")
    print(f"{'#'*60}")

    all_results = {}

    for ds_name in datasets:
        print(f"\n>>> Loading {ds_name}...")
        if ds_name == "SEED":
            subs = list_seed_subs()
            if n_subjects: subs = subs[:n_subjects]
            covs, y = precompute_all_seed(subs, bands=bands, channels=channels)
        elif ds_name == "SEED_IV":
            subs = list_seed_iv_subs()
            if n_subjects: subs = subs[:n_subjects]
            covs, y = precompute_all_seed_iv(subs, bands=bands, channels=channels)
        elif ds_name == "DEAP":
            subs = list_deap_subs()
            if n_subjects: subs = subs[:n_subjects]
            covs, y = precompute_all_deap(subs, bands=bands, channels=channels)
        else:
            print(f"  [SKIP] 未知数据集: {ds_name}")
            continue

        if len(covs) < 2:
            print(f"  [SKIP] {ds_name} 有效被试不足: {len(covs)}")
            continue

        ds_results = {}
        for aligner_name in aligner_names:
            aligner = ALIGNERS[aligner_name]()
            res = evaluate_loso_classification(
                list(covs.keys()), covs, y, aligner,
                bands=bands, classifier="svm", channels=channels
            )
            ds_results[aligner_name] = res
            # 增量保存: 每完成一个 aligner 就 save, 避免中断丢失全部结果
            all_results[ds_name] = ds_results
            if save_result:
                save("T1_baseline", all_results)
                log(f"[T1] incremental save: {ds_name}/{aligner_name} done")

    if save_result:
        save("T1_baseline", all_results)
    return all_results


def _load_latest_t1_results(aligner_names: List[str], datasets: List[str],
                             channels: str = "32ch", bands: str = "5band") -> Dict:
    """从 results/ 加载最近的 T1 结果, 校验配置匹配.

    用于 T5/T6 复用 T1 结果 (方案 2 优化):
        T5 的 32ch+5band 配置与 T1 默认相同
        T6 的 5band+32ch 配置与 T1 默认相同
        两者可直接引用 T1 数值, 无需重算.

    校验条件:
        1. T1 结果文件存在
        2. 包含所有需要的 aligner 和 dataset
        3. 结果的 channels/bands 配置匹配 (从结果字典的元数据推断)

    Returns:
        {ds_name: {aligner: res}} 或 {} (校验失败)
    """
    t1_files = sorted([f for f in os.listdir(RESULTS_DIR)
                       if f.startswith("d1_T1_baseline_")], reverse=True)
    if not t1_files:
        return {}

    fp = os.path.join(RESULTS_DIR, t1_files[0])
    try:
        with open(fp) as f:
            t1_data = json.load(f)
    except Exception:
        return {}

    # 校验: 所有 dataset 和 aligner 都存在
    reused = {}
    for ds_name in datasets:
        if ds_name not in t1_data:
            return {}
        ds_res = t1_data[ds_name]
        for aligner_name in aligner_names:
            if aligner_name not in ds_res:
                return {}
            # 校验结果内的配置 (bands 字段)
            if "bands" in ds_res[aligner_name] and ds_res[aligner_name]["bands"] != bands:
                return {}
            # S2 修复: 强制要求 channels 字段存在且匹配
            # 旧版 T1 结果 (无 channels 字段) 不可信, 拒绝复用, 避免静默通过导致 T5/T6 错配
            if "channels" not in ds_res[aligner_name] or ds_res[aligner_name]["channels"] is None:
                log(f"[T1-reuse] T1 结果缺少 channels 字段 (旧版), 拒绝复用")
                return {}
            if ds_res[aligner_name]["channels"] != channels:
                log(f"[T1-reuse] channels 不匹配: T1={ds_res[aligner_name]['channels']}, 期望={channels}")
                return {}
            # n_subjects 校验: 防止调试版 T1 (如 --n_subjects 3) 被正式 T5/T6 误复用
            # 通过 acc_per_subject 长度推断实际被试数 (T1 结果的 n_subjects 字段不一定可信)
            # None 防护: 若 JSON 中 acc_per_subject 为 null (部分损坏) 或非 list,
            # 必须拒绝复用 (不能用 `or []` 兜底, 否则 None 会静默通过校验)
            # 空列表防护: 当所有 fold 都被单类保护跳过时, acc_list=[] 会写入空列表,
            # 这种 degenerate T1 结果若被复用, 会让 T5/T6 出现 NaN 而无告警, 必须拒绝.
            acc_ps = ds_res[aligner_name].get("acc_per_subject")
            if not isinstance(acc_ps, list):
                log(f"[T1-reuse] T1 结果 acc_per_subject 不是 list "
                    f"(类型={type(acc_ps).__name__}), 拒绝复用")
                return {}
            if len(acc_ps) < 10:
                log(f"[T1-reuse] T1 结果被试数过少 ({len(acc_ps)}), "
                    f"疑似调试版或退化结果 (空列表), 拒绝复用")
                return {}
        reused[ds_name] = {a: t1_data[ds_name][a] for a in aligner_names}

    log(f"[T1-reuse] 复用 {t1_files[0]} (channels={channels}, bands={bands})")
    return reused


# ════════════════════════════════════════════════════════════════════
# T2/T3: 跨库迁移
# ════════════════════════════════════════════════════════════════════

def _unify_labels_to_binary(covs: Dict, y: Dict, dataset_name: str) -> Tuple[Dict, Dict]:
    """将各数据集标签统一为"正vs非正"二分类 (1=正, 0=非正).
    
    标签映射策略 (与 experiment-plan.md 一致):
        SEED:    正(2)→1, 中(1)→0, 负(0)→0
        SEED-IV: happy(3)→1, sad(1)/fear(2)/neutral(0)→0
        DEAP:    高效价(1)→1, 低效价(0)→0 (已是二分类, 无需转换)
    
    Returns:
        covs: 原样返回 (不修改协方差)
        y_new: 统一后的二分类标签
    """
    y_new = {}
    for sub, labels in y.items():
        if dataset_name == "SEED":
            # SEED 统一标签: 2=正, 1=中, 0=负 → 1=正, 0=非正
            y_new[sub] = (labels == 2).astype(np.int64)
        elif dataset_name == "SEED_IV":
            # SEED-IV 标签: 3=happy → 1, 其余 → 0
            y_new[sub] = (labels == 3).astype(np.int64)
        elif dataset_name == "DEAP":
            # DEAP 已是二分类 (1=高效价, 0=低效价), 直接用
            y_new[sub] = labels.astype(np.int64)
        else:
            y_new[sub] = labels.astype(np.int64)
    return covs, y_new


def run_cross_dataset(source_name: str, target_name: str,
                       aligner_names: List[str],
                       channels: str = "32ch", bands: str = "5band",
                       n_subjects: int = None,
                       max_train_epochs_per_subj: int = 100) -> Dict:
    """跨库迁移: source → target.

    标签统一: 所有数据集转为"正vs非正"二分类 (见 _unify_labels_to_binary).
    """
    print(f"\n{'#'*60}")
    print(f"# Cross-Dataset: {source_name} → {target_name}")
    print(f"{'#'*60}")
    
    # 加载源域
    print(f"\n>>> Loading source {source_name}...")
    if source_name == "SEED":
        subs = list_seed_subs()
        if n_subjects: subs = subs[:n_subjects]
        src_covs, src_y = precompute_all_seed(subs, bands=bands, channels=channels)
    elif source_name == "SEED_IV":
        subs = list_seed_iv_subs()
        if n_subjects: subs = subs[:n_subjects]
        src_covs, src_y = precompute_all_seed_iv(subs, bands=bands, channels=channels)
    elif source_name == "DEAP":
        subs = list_deap_subs()
        if n_subjects: subs = subs[:n_subjects]
        src_covs, src_y = precompute_all_deap(subs, bands=bands, channels=channels)
    else:
        print(f"  [SKIP] 未知源域: {source_name}")
        return {}
    
    # 加载目标域
    print(f"\n>>> Loading target {target_name}...")
    if target_name == "SEED":
        subs = list_seed_subs()
        if n_subjects: subs = subs[:n_subjects]
        tgt_covs, tgt_y = precompute_all_seed(subs, bands=bands, channels=channels)
    elif target_name == "DEAP":
        subs = list_deap_subs()
        if n_subjects: subs = subs[:n_subjects]
        tgt_covs, tgt_y = precompute_all_deap(subs, bands=bands, channels=channels)
    elif target_name == "SEED_IV":
        subs = list_seed_iv_subs()
        if n_subjects: subs = subs[:n_subjects]
        tgt_covs, tgt_y = precompute_all_seed_iv(subs, bands=bands, channels=channels)
    else:
        print(f"  [SKIP] 未知目标域: {target_name}")
        return {}
    
    # 标签统一为二分类 (正vs非正)
    src_covs, src_y = _unify_labels_to_binary(src_covs, src_y, source_name)
    tgt_covs, tgt_y = _unify_labels_to_binary(tgt_covs, tgt_y, target_name)
    print(f"  [label] 统一为二分类: source {source_name} dist={np.bincount(np.concatenate(list(src_y.values())), minlength=2)}")
    print(f"  [label] 统一为二分类: target {target_name} dist={np.bincount(np.concatenate(list(tgt_y.values())), minlength=2)}")
    
    all_results = {}
    for aligner_name in aligner_names:
        # ⚠️ S1 修复: 跨数据集场景下 euclidean/riemann/rpa 必须用 reference='grand_mean'
        # 原因: reference='identity' (默认) 下 align_subject 不读取 fit 状态,
        # source/target 各自独立白化到单位阵 I, 不是"target 对齐到 source",
        # 导致跨库迁移实验结论无效. grand_mean 模式下 align_subject 会用 source
        # 的 grand_sqrt_ 把 target 投影到 source 的全局均值, 实现真正的跨域对齐.
        # euclidean 的 identity 模式有同样问题 (align_subject 第 120-128 行),
        # 故三个对齐器统一强制 grand_mean. NoAlignment 无对齐操作, 不受影响.
        if aligner_name in ('euclidean', 'riemann', 'rpa'):
            aligner = ALIGNERS[aligner_name](reference='grand_mean')
            log(f"[S1-fix] {aligner_name}: reference='grand_mean' (跨数据集强制)")
        else:
            aligner = ALIGNERS[aligner_name]()
        res = evaluate_cross_dataset(
            src_covs, src_y, tgt_covs, tgt_y, aligner,
            bands=bands, classifier="svm",
            max_train_epochs_per_subj=max_train_epochs_per_subj,
        )
        all_results[aligner_name] = res

    return all_results


def run_T2(aligner_names, channels="32ch", bands="5band", n_subjects=None):
    """T2: SEED→DEAP."""
    res = run_cross_dataset("SEED", "DEAP", aligner_names, channels, bands, n_subjects)
    save("T2_seed_to_deap", res)
    return res


def run_T3(aligner_names, channels="32ch", bands="5band", n_subjects=None):
    """T3: DEAP→SEED."""
    res = run_cross_dataset("DEAP", "SEED", aligner_names, channels, bands, n_subjects)
    save("T3_deap_to_seed", res)
    return res


def run_T4(aligner_names, channels="32ch", bands="5band", n_subjects=None):
    """T4: SEED↔SEED-IV 同源跨标签 (二分类: 正 vs 非正)."""
    res1 = run_cross_dataset("SEED", "SEED_IV", aligner_names, channels, bands, n_subjects)
    res2 = run_cross_dataset("SEED_IV", "SEED", aligner_names, channels, bands, n_subjects)
    out = {"SEED_to_SEED_IV": res1, "SEED_IV_to_SEED": res2}
    save("T4_seed_seediv", out)
    return out


# ════════════════════════════════════════════════════════════════════
# T4 多类跨标签迁移 (P2-1 新增)
# ════════════════════════════════════════════════════════════════════

# 语义映射到公共 3 类空间 {0=neg, 1=neu, 2=pos}
# SEED 原标签: 0=neg, 1=neu, 2=pos (无需映射)
# SEED-IV 原标签: 0=neutral, 1=sad, 2=fear, 3=happy
#   → 0(neutral)→1(neu), 1(sad)→0(neg), 3(happy)→2(pos), 2(fear)→drop
SEED_IV_TO_COMMON_3CLASS = {0: 1, 1: 0, 3: 2}  # fear(2) 丢弃


def _map_labels_to_common_3class(covs: Dict, y: Dict, dataset_name: str) -> Tuple[Dict, Dict]:
    """将标签映射到公共 3 类空间 {0=neg, 1=neu, 2=pos}, 丢弃无对应样本.

    SEED: 标签已为 {0=neg, 1=neu, 2=pos}, 无需映射.
    SEED-IV: 0(neutral)→1, 1(sad)→0, 3(happy)→2, 2(fear) 丢弃.

    Returns:
        covs_filtered, y_filtered: 仅保留可映射样本
    """
    y_new = {}
    covs_new = {}
    for sub, labels in y.items():
        if dataset_name == "SEED":
            y_new[sub] = labels.astype(np.int64)
            covs_new[sub] = covs[sub]
        elif dataset_name == "SEED_IV":
            # 按 epoch 筛选可映射样本 (label != 2, i.e. 排除 fear)
            mask = labels != 2
            # 映射: 0→1, 1→0, 3→2
            mapped = np.full_like(labels, -1)
            for src, dst in SEED_IV_TO_COMMON_3CLASS.items():
                mapped[labels == src] = dst
            # 仅保留可映射样本 (mapped != -1)
            valid = mapped != -1
            y_new[sub] = mapped[valid].astype(np.int64)
            # covs[sub] 结构: {(lo,hi): (n_epochs, C, C)}, 按 epoch 维度筛选
            covs_new[sub] = {
                band: covs[sub][band][valid] for band in covs[sub]
            }
        else:
            raise ValueError(f"多类跨标签仅支持 SEED↔SEED-IV, 不支持 {dataset_name}")
    return covs_new, y_new


def run_cross_dataset_multiclass(source_name: str, target_name: str,
                                  aligner_names: List[str],
                                  channels: str = "32ch", bands: str = "5band",
                                  n_subjects: int = None) -> Dict:
    """多类跨标签迁移 (3 类公共空间: neg/neu/pos).

    与 run_cross_dataset (二分类) 平行, 但用语义映射 + 样本丢弃, 保留更多类别信息.
    仅支持 SEED↔SEED-IV (同源, 通道一致, 标签可语义映射).
    """
    print(f"\n{'#'*60}")
    print(f"# Multi-Class Cross-Label: {source_name} → {target_name} (3-class)")
    print(f"{'#'*60}")

    # 加载源域
    print(f"\n>>> Loading source {source_name}...")
    if source_name == "SEED":
        subs = list_seed_subs()
        if n_subjects: subs = subs[:n_subjects]
        src_covs, src_y = precompute_all_seed(subs, bands=bands, channels=channels)
    elif source_name == "SEED_IV":
        subs = list_seed_iv_subs()
        if n_subjects: subs = subs[:n_subjects]
        src_covs, src_y = precompute_all_seed_iv(subs, bands=bands, channels=channels)
    else:
        print(f"  [SKIP] 多类跨标签不支持源域: {source_name}")
        return {}

    # 加载目标域
    print(f"\n>>> Loading target {target_name}...")
    if target_name == "SEED":
        subs = list_seed_subs()
        if n_subjects: subs = subs[:n_subjects]
        tgt_covs, tgt_y = precompute_all_seed(subs, bands=bands, channels=channels)
    elif target_name == "SEED_IV":
        subs = list_seed_iv_subs()
        if n_subjects: subs = subs[:n_subjects]
        tgt_covs, tgt_y = precompute_all_seed_iv(subs, bands=bands, channels=channels)
    else:
        print(f"  [SKIP] 多类跨标签不支持目标域: {target_name}")
        return {}

    # 语义映射到公共 3 类空间
    src_covs, src_y = _map_labels_to_common_3class(src_covs, src_y, source_name)
    tgt_covs, tgt_y = _map_labels_to_common_3class(tgt_covs, tgt_y, target_name)
    src_dist = np.bincount(np.concatenate(list(src_y.values())), minlength=3)
    tgt_dist = np.bincount(np.concatenate(list(tgt_y.values())), minlength=3)
    print(f"  [label] 3-class 公共空间: source {source_name} dist(neg/neu/pos)={src_dist}")
    print(f"  [label] 3-class 公共空间: target {target_name} dist(neg/neu/pos)={tgt_dist}")

    all_results = {}
    for aligner_name in aligner_names:
        # ⚠️ S1 修复 (与 run_cross_dataset 二分类版一致): 跨数据集场景下
        # euclidean/riemann/rpa 必须用 reference='grand_mean', 否则
        # align_subject 不读取 fit 状态, source/target 各自独立白化到 I,
        # 多类跨标签迁移实验结论无效.
        if aligner_name in ('euclidean', 'riemann', 'rpa'):
            aligner = ALIGNERS[aligner_name](reference='grand_mean')
            log(f"[S1-fix] {aligner_name}: reference='grand_mean' (跨数据集强制, multiclass)")
        else:
            aligner = ALIGNERS[aligner_name]()
        res = evaluate_cross_dataset(
            src_covs, src_y, tgt_covs, tgt_y, aligner,
            bands=bands, classifier="svm"
        )
        res['label_scheme'] = 'multiclass_3 (neg/neu/pos)'
        all_results[aligner_name] = res

    return all_results


def run_T4_multiclass(aligner_names, channels="32ch", bands="5band", n_subjects=None):
    """T4 多类版: SEED↔SEED-IV 同源跨标签 (3 类语义映射)."""
    res1 = run_cross_dataset_multiclass(
        "SEED", "SEED_IV", aligner_names, channels, bands, n_subjects)
    res2 = run_cross_dataset_multiclass(
        "SEED_IV", "SEED", aligner_names, channels, bands, n_subjects)
    out = {
        "SEED_to_SEED_IV_3class": res1,
        "SEED_IV_to_SEED_3class": res2,
        "label_scheme": "multiclass_3 (neg/neu/pos, fear dropped)",
        "label_mapping": {
            "SEED": "0=neg→0, 1=neu→1, 2=pos→2 (no change)",
            "SEED_IV": "0=neutral→1(neu), 1=sad→0(neg), 3=happy→2(pos), 2=fear→drop",
        },
    }
    save("T4_seed_seediv_multiclass", out)
    return out


# ════════════════════════════════════════════════════════════════════
# T7: Fréchet + CI 量化分析
# ════════════════════════════════════════════════════════════════════

def run_T7(channels: str = "32ch", bands: str = "5band",
           n_subjects: int = None) -> Dict:
    """T7: Fréchet + CI 量化分析."""
    print(f"\n{'#'*60}")
    print(f"# T7: Fréchet + CI Quantitative Analysis")
    print(f"{'#'*60}")
    
    dataset_covs = {}
    
    # 加载所有数据集 (用 NoAlignment, 后续计算 CI)
    for ds_name in ["SEED", "SEED_IV", "DEAP"]:
        print(f"\n>>> Loading {ds_name}...")
        if ds_name == "SEED":
            subs = list_seed_subs()
            if n_subjects: subs = subs[:n_subjects]
            covs, y = precompute_all_seed(subs, bands=bands, channels=channels)
        elif ds_name == "SEED_IV":
            subs = list_seed_iv_subs()
            if n_subjects: subs = subs[:n_subjects]
            covs, y = precompute_all_seed_iv(subs, bands=bands, channels=channels)
        elif ds_name == "DEAP":
            subs = list_deap_subs()
            if n_subjects: subs = subs[:n_subjects]
            covs, y = precompute_all_deap(subs, bands=bands, channels=channels)
        
        if len(covs) > 0:
            dataset_covs[ds_name] = covs
    
    if len(dataset_covs) < 1:
        print("  [SKIP] 无可用数据集")
        return {}
    
    # Fréchet 分析 (含 CI 计算, 对 euclidean / riemann / rpa 对齐器)
    # 注: rpa 已在 frechet_ci.py 注册, 用于验证 H1 (RPA 压缩跨数据集距离)
    # ⚠️ Bug 修复: 之前硬编码 BANDS_5, 当用户传 --bands 8band 时,
    # covs 字典的 key 是 BANDS_8 的细分频段 (如 (8,11), (11,14)),
    # 而 full_frechet_analysis 遍历 BANDS_5 (如 (8,14), (14,31)),
    # 这些 key 在 BANDS_8 中不存在, compute_subject_frechet_matrix 会 KeyError 崩溃.
    # 必须根据 bands 参数动态选择, 与数据加载 (precompute_all_*) 保持一致.
    freq_bands = BANDS_8 if bands == "8band" else BANDS_5
    results = full_frechet_analysis(
        dataset_covs, list(dataset_covs.keys()), freq_bands,
        aligner_names=["euclidean", "riemann", "rpa"],
    )
    
    save("T7_frechet_ci", results)
    return results


# ════════════════════════════════════════════════════════════════════
# T5: 通道敏感性 (32/16/8/4ch)
# ════════════════════════════════════════════════════════════════════

def run_T5(datasets: List[str], aligner_names: List[str],
           bands: str = "5band", n_subjects: int = None) -> Dict:
    """T5: 通道敏感性分析 (32ch → 16ch → 8ch → 4ch).

    方案 2 优化: 32ch+5band 配置与 T1 默认相同, 直接复用 T1 结果, 无需重算.
    仅 16/8/4ch 需要实际计算.
    """
    print(f"\n{'#'*60}")
    print(f"# T5: Channel Sensitivity Analysis")
    print(f"{'#'*60}")

    all_results = {}
    for ch_preset in ["32ch", "16ch", "8ch", "4ch"]:
        print(f"\n>>> Channel preset: {ch_preset}")
        # 32ch+5band 配置与 T1 默认相同, 复用 T1 结果 (方案 2 优化)
        if ch_preset == "32ch" and bands == "5band" and n_subjects is None:
            reused = _load_latest_t1_results(aligner_names, datasets,
                                              channels=ch_preset, bands=bands)
            if reused:
                all_results[ch_preset] = reused
                # 标注来源
                for ds in reused:
                    for a in reused[ds]:
                        all_results[ch_preset][ds][a] = dict(all_results[ch_preset][ds][a])
                        all_results[ch_preset][ds][a]['reused_from'] = 'T1'
                continue
            # 复用失败 (无 T1 结果或配置不匹配), 走正常计算
            log(f"[T5] T1 结果不可用, 重算 32ch")
        # 在每个通道配置上跑 T1 (不保存 T1 结果, 避免覆盖)
        res = run_T1(datasets, aligner_names, channels=ch_preset,
                     bands=bands, n_subjects=n_subjects, save_result=False)
        all_results[ch_preset] = res

    save("T5_channel_sensitivity", all_results)
    return all_results


# ════════════════════════════════════════════════════════════════════
# T6: 频段敏感性
# ════════════════════════════════════════════════════════════════════

def run_T6(datasets: List[str], aligner_names: List[str],
           channels: str = "32ch", n_subjects: int = None) -> Dict:
    """T6: 频段敏感性分析 (5band vs 8band).

    对比 5 频段 (δ/θ/α/β/γ) 和 8 频段 (细分) 配置下的分类性能.

    方案 2 优化: 5band+32ch 配置与 T1 默认相同, 直接复用 T1 结果, 无需重算.
    仅 8band 需要实际计算.
    """
    log("T6: Band Sensitivity Analysis starting")
    print(f"\n{'#'*60}")
    print(f"# T6: Band Sensitivity Analysis (5band vs 8band)")
    print(f"{'#'*60}")

    all_results = {}
    for bands_config in ["5band", "8band"]:
        log(f"T6: running with {bands_config}")
        print(f"\n>>> Bands config: {bands_config}")
        # 5band+32ch 配置与 T1 默认相同, 复用 T1 结果 (方案 2 优化)
        if bands_config == "5band" and channels == "32ch" and n_subjects is None:
            reused = _load_latest_t1_results(aligner_names, datasets,
                                              channels=channels, bands=bands_config)
            if reused:
                all_results[bands_config] = reused
                # 标注来源
                for ds in reused:
                    for a in reused[ds]:
                        all_results[bands_config][ds][a] = dict(all_results[bands_config][ds][a])
                        all_results[bands_config][ds][a]['reused_from'] = 'T1'
                continue
            # 复用失败, 走正常计算
            log(f"[T6] T1 结果不可用, 重算 5band")
        res = run_T1(datasets, aligner_names, channels=channels,
                     bands=bands_config, n_subjects=n_subjects, save_result=False)
        all_results[bands_config] = res

    save("T6_band_sensitivity", all_results)
    log("T6: done")
    return all_results


# ════════════════════════════════════════════════════════════════════
# T8: SOTA 对比
# ════════════════════════════════════════════════════════════════════

def _run_itsa_baseline(covs: Dict, y: Dict,
                       bands: List, n_features: int = 150,
                       max_train_epochs_per_subj: int = 50) -> Dict:
    """ITSA baseline: 切空间对齐 (无黎曼对齐, 仅 TS+ SVM).

    ITSA (Lai-Tan 2025) 的核心是切空间对齐.
    这里实现其简化版: 无 SPD 对齐, 直接在切空间 + SVM.
    与我们的 Riemannian Alignment + TS + SVM 形成对比.

    优化: 限制每被试训练 epoch 数, 避免 TangentSpace 对海量矩阵计算过慢.
    45 被试 × 831 epochs = 36000+ epochs, TangentSpace fit_transform 很慢.
    限制为 50 epochs/被试 → 2200 epochs, 速度提升 ~16 倍.
    """
    from pyriemann.tangentspace import TangentSpace
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    subjects = list(covs.keys())
    acc_list, f1_list = [], []
    all_y_true, all_y_pred = [], []

    for i, test_subj in enumerate(subjects):
        t_fold = time.time()
        train_subjs = [s for s in subjects if s != test_subj]

        features_list, test_features_list = [], []
        for band in bands:
            # 限制每被试训练 epoch 数, 加速 TangentSpace
            # 均匀采样而非取前 N 个 (SEED 标签按顺序排列, 前 N 个可能同类)
            train_covs_list = []
            train_labels_list = []
            for s in train_subjs:
                n_total = len(covs[s][band])
                n_ep = min(n_total, max_train_epochs_per_subj)
                if n_total > n_ep:
                    # 均匀采样
                    idx = np.linspace(0, n_total - 1, n_ep, dtype=int)
                else:
                    idx = np.arange(n_total)
                train_covs_list.append(covs[s][band][idx])
                train_labels_list.append(y[s][idx])

            train_covs_pooled = np.concatenate(train_covs_list, axis=0)
            train_labels_pooled = np.concatenate(train_labels_list)

            ts = TangentSpace(metric='riemann')
            ts_feats = ts.fit_transform(train_covs_pooled, train_labels_pooled)
            features_list.append(ts_feats)

            test_covs = covs[test_subj][band]
            test_feats = ts.transform(test_covs)
            test_features_list.append(test_feats)

        X_train = np.hstack(features_list)
        X_test = np.hstack(test_features_list)
        y_train = train_labels_pooled
        y_test = y[test_subj]

        # 单类保护: 训练集只有一类时, SelectKBest(f_classif) 和 SVC 会崩溃
        # 小数据集 (n_subjects=2 调试模式) 或极端采样可能触发
        if len(np.unique(y_train)) < 2:
            print(f"    [{i+1}/{len(subjects)}] {test_subj}: "
                  f"SKIP (训练集仅含单类)", flush=True)
            continue

        if X_train.shape[1] > n_features:
            sel = SelectKBest(f_classif, k=n_features)
            X_train = sel.fit_transform(X_train, y_train)
            X_test = sel.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = SVC(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        acc_list.append(acc)
        f1_list.append(f1)
        all_y_true.extend(list(y_test))
        all_y_pred.extend(list(y_pred))
        print(f"    [{i+1}/{len(subjects)}] {test_subj}: "
              f"ACC={acc:.4f}, F1={f1:.4f} ({time.time()-t_fold:.1f}s)", flush=True)

    # 若所有 fold 都因单类被跳过, 返回空结果而非崩溃
    if len(acc_list) == 0:
        return {
            'acc_mean': 0.0, 'acc_std': 0.0,
            'f1_mean': 0.0, 'f1_std': 0.0,
            'n_subjects': len(subjects),
            'n_valid_folds': 0,
            'max_train_epochs_per_subj': max_train_epochs_per_subj,
            'note': '所有 fold 因训练集单类被跳过',
        }

    labels_sorted = sorted(set(all_y_true) | set(all_y_pred))
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels_sorted)

    return {
        'acc_mean': float(np.mean(acc_list)),
        'acc_std': float(np.std(acc_list)),
        'acc_per_subject': [float(x) for x in acc_list],
        'f1_mean': float(np.mean(f1_list)),
        'f1_std': float(np.std(f1_list)),
        'f1_per_subject': [float(x) for x in f1_list],
        'n_subjects': len(subjects),
        'n_valid_folds': len(acc_list),
        'aligner': 'none',
        'bands': f"{len(bands)}band",
        'classifier': 'svm',
        'n_features': n_features,
        'max_train_epochs_per_subj': max_train_epochs_per_subj,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': [int(x) for x in labels_sorted],
    }


def run_T8(datasets: List[str], channels: str = "32ch",
           bands: str = "5band", n_subjects: int = None,
           dl_baselines: List[str] = None) -> Dict:
    """T8: SOTA 对比.

    本地可跑的 baseline:
        - EA (He 2018): 欧氏对齐 — 已在 T1 中跑过, 此处引用
        - ITSA (Lai-Tan 2025): 切空间对齐 — 本函数实现
        - Ours (Riemannian): 黎曼对齐 — 已在 T1 中跑过, 此处引用
        - DGCNN-SPD (Song 2018, SPD 变体): 本地训练, dl_baselines=['DGCNN-SPD'] 时启用
        - mdJPT (Liu 2025 NeurIPS): 本地训练 (需 MDJPT_ROOT 环境变量, 否则用 fallback)

    深度学习 baseline (引用文献数据, 标记 †):
        - SF-UDA (Imtiaz 2026): Source-Free UDA
        - PAA (Li 2026): 对抗域适应
        - Zhang 2024 TETCI: 时空黎曼

    注: 深度学习方法需要在 GPU 上训练, 且原始代码/数据划分不完全一致,
        故引用论文报告值, 在表格中标注 †.

    Args:
        dl_baselines: 本地训练的 DL baseline 名称列表, 如 ['DGCNN-SPD', 'mdJPT'].
            None 或空列表=不跑 DL baseline (仅 ITSA + 文献数值).
    """
    print(f"\n{'#'*60}")
    print(f"# T8: SOTA Comparison")
    print(f"{'#'*60}")

    freq_bands = BANDS_5 if bands == "5band" else BANDS_8
    all_results = {}

    # ── fail-fast: 先检查 T1 结果是否存在 (避免跑完 ITSA ~26min 才发现 T1 缺失) ──
    # T8 引用 T1 的 EA 和 RPA 结果, 若 T1 缺失应立即报错, 不浪费 ITSA 计算时间.
    # 调试模式 (n_subjects is not None): 若 T1 结果是被试数过少的调试版, 仍允许继续,
    # 但打印警告, 供用户验证 T8 流程; 正式模式 (n_subjects is None): 严格拒绝.
    print(f"\n>>> Checking T1 results availability (fail-fast before ITSA)...")
    t1_reused = _load_latest_t1_results(
        ['euclidean', 'rpa', 'riemann'], datasets, channels=channels, bands=bands
    )
    if not t1_reused:
        if n_subjects is not None:
            print(f"    [WARNING] 调试模式: T1 结果不可复用 (可能被试数过少或配置不匹配), "
                  f"将重算 T1 作为 T8 的 EA/RPA 来源")
            # 调试模式 fallback: 重算 T1
            t1_reused = run_T1(datasets, ['euclidean', 'rpa', 'riemann'],
                               channels=channels, bands=bands,
                               n_subjects=n_subjects, save_result=False)
        else:
            raise RuntimeError(
                f"T8 需要 T1 结果, 但未找到匹配的 T1 结果文件 "
                f"(channels={channels}, bands={bands}). 请先运行 T1."
            )
    for ds_name in datasets:
        if ds_name not in t1_reused:
            raise RuntimeError(
                f"T8 需要 {ds_name} 的 T1 结果, 但未找到. 请先运行 T1."
            )
    print(f"    [OK] T1 结果已就绪, 将在 ITSA 后提取 EA/RPA 数值")

    # ── 本地跑 ITSA ──
    for ds_name in datasets:
        print(f"\n>>> Loading {ds_name} for ITSA baseline...")
        if ds_name == "SEED":
            subs = list_seed_subs()
            if n_subjects: subs = subs[:n_subjects]
            covs, y = precompute_all_seed(subs, bands=bands, channels=channels)
        elif ds_name == "SEED_IV":
            subs = list_seed_iv_subs()
            if n_subjects: subs = subs[:n_subjects]
            covs, y = precompute_all_seed_iv(subs, bands=bands, channels=channels)
        elif ds_name == "DEAP":
            subs = list_deap_subs()
            if n_subjects: subs = subs[:n_subjects]
            covs, y = precompute_all_deap(subs, bands=bands, channels=channels)
        else:
            continue

        if len(covs) < 2:
            continue

        print(f"  Running ITSA baseline ({len(covs)} subs)...")
        t0 = time.time()
        itsa_res = _run_itsa_baseline(covs, y, freq_bands)
        itsa_res['method'] = 'ITSA'
        itsa_res['time_s'] = time.time() - t0
        print(f"  ITSA: ACC={itsa_res['acc_mean']:.4f}±{itsa_res['acc_std']:.4f} "
              f"({itsa_res['time_s']:.1f}s)")

        all_results.setdefault(ds_name, {})['ITSA'] = itsa_res

        # ── 本地 DL baselines (复用已加载的 covs/y) ──
        if dl_baselines:
            for bl_name in dl_baselines:
                if bl_name.upper() in ('NONE', ''):
                    continue
                print(f"\n  Running DL baseline: {bl_name} ({len(covs)} subs)...")
                try:
                    from baselines import get_baseline
                    bl = get_baseline(bl_name)
                    t0 = time.time()
                    bl_res = bl.evaluate_loso(
                        list(covs.keys()), covs, y, freq_bands,
                        verbose=True
                    )
                    bl_res['time_s'] = time.time() - t0
                    print(f"  {bl_name}: ACC={bl_res['acc_mean']:.4f}±"
                          f"{bl_res['acc_std']:.4f} ({bl_res['time_s']:.1f}s)")
                    all_results[ds_name][bl_name] = bl_res
                    # 增量保存, 避免中断丢失
                    save("T8_sota_comparison", all_results)
                except Exception as e:
                    print(f"  [WARNING] {bl_name} 失败: {e}")
                    all_results[ds_name][bl_name] = {
                        'error': str(e),
                        'method': bl_name,
                    }
                    save("T8_sota_comparison", all_results)

    # ── 深度学习 baseline (引用文献数值, 全部已核实) ──
    # 按 AGENTS.md 规则, 仅填入经核实原论文的数值.
    # 数值来源 (已查阅原论文 PDF):
    #   1. RGNN (Zhong et al. 2020 IEEE TAFFC)
    #      arXiv:1907.07835, 作者 Peixiang Zhong, Di Wang, Chunyan Miao
    #      SEED subject-independent (Table 2, all bands): 85.30 ± 06.72
    #      SEED-IV subject-independent (Table 2, all bands): 73.84 ± 08.02
    #      DEAP: 原论文未在 DEAP 上做实验
    #   2. Grop (Wu et al. 2024 IEEE TAFFC, vol. 16, no. 1, pp. 324-336)
    #      IEEE Xplore document 10609541, 第一作者 吴梦琪, 通讯作者 张通 (华南理工大学)
    #      SEED cross-subject LOSO (Table II, all bands): 91.58 ± 4.02
    #      SEED-IV cross-subject LOSO (Table IV, all bands): 75.63 ± 09.20
    #      DEAP: 原论文未在 DEAP 上做实验
    literature_results = {
        "SEED": {
            "RGNN†": {
                "acc_mean": 0.8530,
                "acc_std": 0.0672,
                "note": "Zhong 2020 IEEE TAFFC, subject-independent LOSO, all bands, 62ch (原论文) vs 32ch (本项目)",
                "source": "arXiv:1907.07835 Table 2",
                "method": "RGNN (Zhong 2020)",
                "channels_reported": "62ch",
                "channels_compared": channels,
            },
            "Grop†": {
                "acc_mean": 0.9158,
                "acc_std": 0.0402,
                "note": "Wu 2024 IEEE TAFFC, cross-subject LOSO, all bands, 62ch (原论文) vs 32ch (本项目)",
                "source": "IEEE Xplore document 10609541 Table II",
                "method": "Grop (Wu 2024)",
                "channels_reported": "62ch",
                "channels_compared": channels,
            },
            # M2 修复: PAA 是跨语料库 LOSO (Protocol 3), 与单数据集 T8 表格口径不一致,
            # 已移到 _cross_dataset_literature, 避免误导 reviewer
        },
        "SEED_IV": {
            "RGNN†": {
                "acc_mean": 0.7384,
                "acc_std": 0.0802,
                "note": "Zhong 2020 IEEE TAFFC, subject-independent LOSO, all bands, 62ch (原论文) vs 32ch (本项目)",
                "source": "arXiv:1907.07835 Table 2",
                "method": "RGNN (Zhong 2020)",
                "channels_reported": "62ch",
                "channels_compared": channels,
            },
            "Grop†": {
                "acc_mean": 0.7563,
                "acc_std": 0.0920,
                "note": "Wu 2024 IEEE TAFFC, cross-subject LOSO, all bands, 62ch (原论文) vs 32ch (本项目)",
                "source": "IEEE Xplore document 10609541 Table IV",
                "method": "Grop (Wu 2024)",
                "channels_reported": "62ch",
                "channels_compared": channels,
            },
        },
        "DEAP": {
            # RGNN 和 Grop 原论文均未在 DEAP 上做实验
            # SF-UDA 做了 SEED↔DEAP 跨库迁移, 但那是跨数据集数值, 不是单数据集 LOSO
            # 故 DEAP 单数据集列不填入文献数值
        },
    }

    # ── 跨数据集迁移文献数值 (SF-UDA, 用于 T2/T3 对比) ──
    # SF-UDA (Imtiaz 2026) 做了 SEED↔DEAP 双向迁移, 与本项目 T2/T3 直接可比
    # 来源: arXiv:2606.28202v1, PDF Table 1-3, page 9
    cross_dataset_literature = {
        "SEED_to_DEAP": {
            "SF-UDA†": {
                "acc_mean": 0.6138,
                "acc_std": None,  # PDF 未报告 std
                "note": "Imtiaz 2026, SEED→DEAP 二分类 (正vs非正), 32ch 公共子集",
                "source": "arXiv:2606.28202v1 Table 1, page 9 (binary classification)",
                "method": "SF-UDA (Imtiaz 2026)",
            },
        },
        "DEAP_to_SEED": {
            "SF-UDA†": {
                "acc_mean": 0.6956,
                "acc_std": None,
                "note": "Imtiaz 2026, DEAP→SEED 二分类 (正vs非正), 32ch 公共子集",
                "source": "arXiv:2606.28202v1 Table 1, page 9 (binary classification)",
                "method": "SF-UDA (Imtiaz 2026)",
            },
        },
    }

    # 统一标注文献核实状态 (符合 AGENTS.md "不虚构引用" 规则)
    _verified_fields = {"verified": True, "verified_date": "2026-07-14",
                        "verified_by": "PDF 提取"}
    for ds_lit in list(literature_results.values()):
        for entry in ds_lit.values():
            entry.update(_verified_fields)
            entry["acc_std_reported"] = entry.get("acc_std") is not None
    for ds_lit in list(cross_dataset_literature.values()):
        for entry in ds_lit.values():
            entry.update(_verified_fields)
            entry["acc_std_reported"] = entry.get("acc_std") is not None

    for ds_name, lit_res in literature_results.items():
        if ds_name in all_results:
            all_results[ds_name].update(lit_res)
        else:
            all_results[ds_name] = dict(lit_res)

    # 把跨数据集文献数值也存入 all_results
    all_results['_cross_dataset_literature'] = cross_dataset_literature

    print(f"\n>>> [NOTE] SOTA baseline 已全部核实并填入 (2026-07-14 PDF 提取):")
    print(f"    单数据集 LOSO:")
    print(f"    - RGNN (Zhong 2020): SEED=0.8530±0.0672, SEED-IV=0.7384±0.0802 (Table 2)")
    print(f"    - Grop (Wu 2024): SEED=0.9158±0.0402, SEED-IV=0.7563±0.0920 (Table II/IV)")
    print(f"    跨数据集迁移 (与 T2/T3 对比):")
    print(f"    - SF-UDA (Imtiaz 2026): SEED→DEAP=0.6138, DEAP→SEED=0.6956 (Table 1, binary)")
    print(f"    - 注: SF-UDA 同样用 32ch 公共子集, 与本项目 T2/T3 直接可比")
    print(f"    - 注: PAA (Li 2026) 是跨库 LOSO 协议 (Protocol 3), 与 T1 单库 LOSO 不可比, 已从单数据集表移除")

    # ── 从 T1 结果提取 EA 和 RPA baseline ──
    # T1 存在性检查已在函数开头 fail-fast 完成, 这里仅提取数值
    print(f"\n>>> Extracting EA and RPA baselines from T1 results...")
    for ds_name in datasets:
        # EA (euclidean alignment)
        if 'euclidean' in t1_reused[ds_name]:
            ea_res = t1_reused[ds_name]['euclidean']
            all_results.setdefault(ds_name, {})['EA'] = {
                'acc_mean': ea_res['acc_mean'],
                'acc_std': ea_res['acc_std'],
                'f1_mean': ea_res['f1_mean'],
                'f1_std': ea_res['f1_std'],
                'method': 'EA (He 2018)',
                'note': '从 T1 结果引用',
            }
        # Ours: 优先 RPA, fallback riemann (键名改为 Ours_fallback 避免误导)
        if 'rpa' in t1_reused[ds_name]:
            rpa_res = t1_reused[ds_name]['rpa']
            all_results.setdefault(ds_name, {})['Ours'] = {
                'acc_mean': rpa_res['acc_mean'],
                'acc_std': rpa_res['acc_std'],
                'f1_mean': rpa_res['f1_mean'],
                'f1_std': rpa_res['f1_std'],
                'method': 'RPA (Rodrigues 2019)',
                'note': '从 T1 结果引用 (核心方法: Riemannian Procrustes Alignment)',
            }
        elif 'riemann' in t1_reused[ds_name]:
            rm_res = t1_reused[ds_name]['riemann']
            all_results.setdefault(ds_name, {})['Ours_fallback'] = {
                'acc_mean': rm_res['acc_mean'],
                'acc_std': rm_res['acc_std'],
                'f1_mean': rm_res['f1_mean'],
                'f1_std': rm_res['f1_std'],
                'method': 'Riemannian Alignment (Zanini 2018) [fallback]',
                'note': 'T1 未跑 rpa, 暂用 riemann (键名 Ours_fallback), 待重跑 T1 含 rpa',
            }

    save("T8_sota_comparison", all_results)
    return all_results


# ════════════════════════════════════════════════════════════════════
# T9: 统计显著性检验 (P0-3 新增)
# ════════════════════════════════════════════════════════════════════

def run_T9(datasets: List[str], source_table: str = "T1",
           n_permutations: int = 10000) -> Dict:
    """T9: 对已有 T1/T2/T3/T4 结果做配对统计检验.

    读取最近的 source_table 结果, 对每个数据集内的多个对齐器 vs 'none' baseline
    做 paired t-test (单侧 greater) + permutation test (双侧).

    Args:
        datasets: 要分析的数据集列表
        source_table: 'T1' | 'T2' | 'T3' | 'T4' (从哪个表的结果读)
        n_permutations: 置换次数

    Returns:
        dict: {per_dataset: {...}, pooled_across_datasets: {...}}
    """
    print(f"\n{'#'*60}")
    print(f"# T9: Statistical Significance Tests")
    print(f"# Source: {source_table}, Datasets: {datasets}")
    print(f"# Permutations: {n_permutations}")
    print(f"{'#'*60}")

    # 找到最近的 source_table 结果文件
    if source_table == "T4":
        # T4 结果结构是嵌套的 {SEED_to_SEED_IV: {aligner:...}, SEED_IV_to_SEED:...}
        # 注意: pattern "d1_T4_seed_seediv_" 也会匹配 T4mc 文件 (d1_T4_seed_seediv_multiclass_*)
        # 必须排除 multiclass 后缀, 否则 T9 --stats_source T4 可能默默拿到 T4mc 的 3 类统计
        pattern = "d1_T4_seed_seediv_"
        exclude_substring = "multiclass"
    else:
        # ⚠️ Bug 修复: 之前用 source_table.lower() 产生 "d1_t2_", 但实际保存的文件名是
        # 大写 (run_T2 调用 save("T2_seed_to_deap", ...) 产生 d1_T2_seed_to_deap_<ts>.json).
        # 用 lower() 会找不到任何文件, T9 默默返回空. 必须保留大写.
        pattern = f"d1_{source_table}_"
        exclude_substring = None
    files = sorted(
        [f for f in os.listdir(RESULTS_DIR)
         if f.startswith(pattern) and (exclude_substring is None or exclude_substring not in f)],
        reverse=True
    )
    if not files:
        print(f"  [ERROR] 未找到 {source_table} 结果文件 (pattern: {pattern})")
        return {}

    result_file = os.path.join(RESULTS_DIR, files[0])
    print(f"  [load] {result_file}")
    with open(result_file) as f:
        table_data = json.load(f)

    all_stats = {}

    if source_table == "T4":
        # T4: 对每个方向做统计
        for direction, dir_results in table_data.items():
            if not isinstance(dir_results, dict):
                continue
            # dir_results = {aligner_name: {acc_per_subject: [...], ...}}
            # 只处理含 acc_per_subject 的 aligner
            aligner_results = {
                k: v for k, v in dir_results.items()
                if isinstance(v, dict) and "acc_per_subject" in v
            }
            if not aligner_results:
                continue
            print(f"\n>>> Stats for T4/{direction}:")
            stats = compare_aligners(aligner_results, reference="none",
                                     n_permutations=n_permutations)
            all_stats[direction] = stats
            for name, s in stats.items():
                print(f"  {name} vs none: Δ={s['delta_mean']:+.4f}, "
                      f"t={s['t_test']['t_stat']:+.3f} (p={s['t_test']['p_value']:.4f}), "
                      f"perm p={s['permutation_test']['p_value']:.4f}")
    elif source_table in ("T2", "T3"):
        # T2/T3 结果结构是 {aligner: res} (无 dataset 外层), 与 T1 的 {ds: {aligner: res}} 不同
        # 直接把 table_data 当作单数据集的 aligner_results 处理
        aligner_results = {
            k: v for k, v in table_data.items()
            if isinstance(v, dict) and "acc_per_subject" in v
        }
        if not aligner_results:
            print(f"  [SKIP] {source_table} 无 aligner 结果 (含 acc_per_subject)")
        else:
            print(f"\n>>> Stats for {source_table}:")
            stats = compare_aligners(aligner_results, reference="none",
                                     n_permutations=n_permutations)
            all_stats[source_table] = stats
            for name, s in stats.items():
                print(f"  {name} vs none: Δ={s['delta_mean']:+.4f}, "
                      f"t={s['t_test']['t_stat']:+.3f} (p={s['t_test']['p_value']:.4f}), "
                      f"perm p={s['permutation_test']['p_value']:.4f}")
    else:
        # T1: 对每个数据集做统计
        for ds_name in datasets:
            if ds_name not in table_data:
                print(f"  [SKIP] {ds_name} 不在 {source_table} 结果中")
                continue
            ds_results = table_data[ds_name]
            # 只处理含 acc_per_subject 的 aligner
            aligner_results = {
                k: v for k, v in ds_results.items()
                if isinstance(v, dict) and "acc_per_subject" in v
            }
            if not aligner_results:
                print(f"  [SKIP] {ds_name} 无 aligner 结果")
                continue
            print(f"\n>>> Stats for {source_table}/{ds_name}:")
            stats = compare_aligners(aligner_results, reference="none",
                                     n_permutations=n_permutations)
            all_stats[ds_name] = stats
            for name, s in stats.items():
                print(f"  {name} vs none: Δ={s['delta_mean']:+.4f}, "
                      f"t={s['t_test']['t_stat']:+.3f} (p={s['t_test']['p_value']:.4f}), "
                      f"perm p={s['permutation_test']['p_value']:.4f}")

        # 跨数据集聚合 (仅 T1 有多数据集)
        if source_table == "T1" and len(all_stats) > 1:
            print(f"\n>>> Pooled stats across datasets (vs none):")
            # 重建 {ds: {aligner: {acc_per_subject}}} 结构给 compare_all_datasets
            pooled_input = {}
            for ds_name in datasets:
                if ds_name not in table_data:
                    continue
                ds_res = table_data[ds_name]
                pooled_input[ds_name] = {
                    k: v for k, v in ds_res.items()
                    if isinstance(v, dict) and "acc_per_subject" in v
                }
            pooled_stats = compare_all_datasets(pooled_input)
            all_stats['_pooled_across_datasets'] = pooled_stats
            for name, s in pooled_stats.items():
                print(f"  {name} vs none (pooled): Δ={s['mean_delta']:+.4f}, "
                      f"t={s['t_stat']:+.3f} (p={s['p_value']:.4f}), n={s['n_total']}")

    save("T9_statistics", all_stats)
    return all_stats


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="D1 Experiments")
    parser.add_argument("--tables", type=str, default="T1",
                        help="实验表, 逗号分隔 (T1,T2,T3,T4,T4mc,T5,T6,T7,T8,T9)")
    parser.add_argument("--datasets", type=str, default="SEED,SEED_IV,DEAP",
                        help="数据集, 逗号分隔")
    parser.add_argument("--aligners", type=str, default="none,euclidean,riemann,rpa",
                        help="对齐器, 逗号分隔")
    parser.add_argument("--channels", type=str, default="32ch",
                        help="通道配置 (32ch/16ch/8ch/4ch)")
    parser.add_argument("--bands", type=str, default="5band",
                        help="频段配置 (5band/8band)")
    parser.add_argument("--n_subjects", type=int, default=None,
                        help="每数据集最多用前 N 个被试 (调试用)")
    parser.add_argument("--stats_source", type=str, default="T1",
                        help="T9 统计检验的数据源 (T1/T2/T3/T4)")
    parser.add_argument("--n_permutations", type=int, default=10000,
                        help="T9 置换检验次数 (默认 10000)")
    parser.add_argument("--dl_baselines", type=str, default="",
                        help="T8 本地训练的 DL baseline, 逗号分隔 "
                             "(DGCNN-SPD, mdJPT). 空=不跑 DL baseline")
    args = parser.parse_args()
    
    # ⚠️ Bug 修复: 之前用 .split(",") 不 strip, 用户输入 "SEED, SEED_IV" 时
    # 产生 ["SEED", " SEED_IV"], 后续 if ds_name == "SEED_IV" 不匹配, 静默跳过.
    # --dl_baselines 已做 strip, 这里统一处理.
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    aligners = [a.strip() for a in args.aligners.split(",") if a.strip()]
    
    print(f"\n{'='*60}")
    print(f"D1 Experiments")
    print(f"  Tables: {tables}")
    print(f"  Datasets: {datasets}")
    print(f"  Aligners: {aligners}")
    print(f"  Channels: {args.channels}, Bands: {args.bands}")
    if args.n_subjects:
        print(f"  N_subjects: {args.n_subjects} (debug mode)")
    print(f"{'='*60}")
    
    t_start = time.time()
    
    if "T1" in tables:
        log("=== T1 starting ===")
        run_T1(datasets, aligners, args.channels, args.bands, args.n_subjects)
        log("=== T1 done ===")

    if "T2" in tables:
        log("=== T2 starting ===")
        run_T2(aligners, args.channels, args.bands, args.n_subjects)
        log("=== T2 done ===")

    if "T3" in tables:
        log("=== T3 starting ===")
        run_T3(aligners, args.channels, args.bands, args.n_subjects)
        log("=== T3 done ===")

    if "T4" in tables:
        log("=== T4 starting ===")
        run_T4(aligners, args.channels, args.bands, args.n_subjects)
        log("=== T4 done ===")

    if "T4mc" in tables:
        log("=== T4 multiclass starting ===")
        run_T4_multiclass(aligners, args.channels, args.bands, args.n_subjects)
        log("=== T4 multiclass done ===")

    if "T5" in tables:
        log("=== T5 starting ===")
        run_T5(datasets, aligners, args.bands, args.n_subjects)
        log("=== T5 done ===")

    if "T6" in tables:
        log("=== T6 starting ===")
        run_T6(datasets, aligners, args.channels, args.n_subjects)
        log("=== T6 done ===")

    if "T7" in tables:
        log("=== T7 starting ===")
        run_T7(args.channels, args.bands, args.n_subjects)
        log("=== T7 done ===")

    if "T8" in tables:
        log("=== T8 starting ===")
        # 解析 DL baseline 列表
        dl_baselines = [b.strip() for b in args.dl_baselines.split(",")
                         if b.strip()] if args.dl_baselines else None
        run_T8(datasets, args.channels, args.bands, args.n_subjects,
               dl_baselines=dl_baselines)
        log("=== T8 done ===")

    if "T9" in tables:
        log("=== T9 statistics starting ===")
        # 默认对 T1 结果做统计; 可通过 --stats_source 覆盖
        stats_source = getattr(args, "stats_source", "T1")
        run_T9(datasets, source_table=stats_source,
               n_permutations=args.n_permutations)
        log("=== T9 done ===")
    
    print(f"\n{'='*60}")
    print(f"All done. Total time: {time.time()-t_start:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
