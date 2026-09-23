"""
cov_cache.py — 协方差矩阵磁盘缓存
==================================

避免 T1/T2/T3/T4/T7 实验重复计算协方差矩阵.
完整实验中, SEED 加载 4 次, SEED_IV 3 次, DEAP 4 次,
缓存可将数据加载时间从 ~40min 降到 ~5min (仅首次计算).

缓存策略:
    - 首次计算 precompute_subject_covs 时, 保存到 cache/ 目录
    - 后续相同参数的调用直接从磁盘加载
    - 缓存 key 包含所有影响输出的参数
    - 用 .npz (压缩) 格式, 单被试单文件

缓存位置: 项目根/cache/
    文件名: {dataset}_{subject}_{params_hash}.npz

用法 (在 loader 中):
    from cov_cache import cached_precompute
    cov_dict, y, n_ch = cached_precompute(dataset, subject, compute_fn, **params)
"""
from __future__ import annotations

import hashlib
import os
from typing import Callable, Dict, Tuple

import numpy as np

# 缓存目录: 项目根/cache/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_PROJECT_ROOT, "cache")

# 缓存版本号: 当 BANDS_8 / 通道配置 / 标签映射等发生破坏性变更时递增,
# 使旧缓存自动失效 (hash 不同 → 文件名不同 → 重新计算).
# 历史:
#   v1 (2026-07-08): 初始版本
#   v2 (2026-07-14): BANDS_8 从非标准分解改为标准分解 (δ/θ/low-α/high-α/low-β/high-β/low-γ/high-γ)
#                    + WEARABLE_16CH 补顶叶通道
#   v3 (2026-07-21): DEAP '32ch' 从原生顺序改为 COMMON_32CH 字母序,
#                    与 SEED/SEED-IV 通道顺序对齐. 旧 DEAP 缓存必须失效重算,
#                    否则 T2/T3 跨库迁移协方差行列对应不同通道.
CACHE_VERSION = "v3"


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _params_hash(dataset: str, subject: str, **params) -> str:
    """生成参数哈希 (短, 用于文件名)."""
    # 按固定顺序排列参数, 保证哈希稳定
    keys = sorted(params.keys())
    parts = [CACHE_VERSION, dataset, subject]
    for k in keys:
        v = params[k]
        parts.append(f"{k}={v}")
    s = "|".join(str(p) for p in parts)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _cache_path(dataset: str, subject: str, **params) -> str:
    """生成缓存文件路径."""
    h = _params_hash(dataset, subject, **params)
    # subject 可能含 / (如 SEED-IV 的 s1/10_20151014), 替换为 _
    safe_subj = subject.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{dataset}_{safe_subj}_{h}.npz")


def _save_cache(path: str, cov_dict: Dict, y: np.ndarray, n_ch: int):
    """保存协方差到缓存文件."""
    _ensure_cache_dir()
    data = {"y": y, "n_ch": np.array(n_ch)}
    for (lo, hi), cov in cov_dict.items():
        data[f"band_{lo}_{hi}"] = cov
    np.savez_compressed(path, **data)


def _load_cache(path: str) -> Tuple[Dict, np.ndarray, int]:
    """从缓存文件加载协方差."""
    data = np.load(path, allow_pickle=False)
    y = data["y"]
    n_ch = int(data["n_ch"])
    cov_dict = {}
    for key in data.files:
        if key.startswith("band_"):
            parts = key.split("_")
            band = (float(parts[1]), float(parts[2]))
            cov_dict[band] = data[key]
    return cov_dict, y, n_ch


def cached_precompute(
    dataset: str,
    subject: str,
    compute_fn: Callable,
    use_cache: bool = True,
    **params,
) -> Tuple[Dict, np.ndarray, int]:
    """带缓存的协方差预计算.

    Args:
        dataset: 数据集名 (SEED / SEED_IV / DEAP)
        subject: 被试 ID
        compute_fn: 实际计算函数 (如 precompute_subject_covs)
        use_cache: 是否使用缓存 (False 则强制重算)
        **params: 传给 compute_fn 的参数 (影响输出, 用于缓存 key)

    Returns:
        cov_dict: {(lo, hi): (n_epochs, C, C)}
        y: (n_epochs,) 标签
        n_ch: 通道数
    """
    if not use_cache:
        return compute_fn(subject, **params)

    path = _cache_path(dataset, subject, **params)

    # 尝试加载缓存
    if os.path.exists(path):
        try:
            return _load_cache(path)
        except Exception:
            pass  # 缓存损坏, 重新计算

    # 计算 + 保存
    result = compute_fn(subject, **params)
    try:
        _save_cache(path, *result)
    except Exception:
        pass  # 缓存写入失败不影响运行

    return result


def clear_cache(dataset: str = None):
    """清空缓存.

    Args:
        dataset: 指定数据集 (None=全部)

    注意: 数据集名前缀匹配需小心 "SEED" vs "SEED_IV".
        朴素用 `fn.startswith("SEED_")` 会误删 SEED_IV 文件,
        因为 "SEED_IV_xxx.npz" 也以 "SEED_" 开头.
        修复: 优先匹配更长的前缀 "SEED_IV_", 再匹配 "SEED_".
    """
    if not os.path.isdir(CACHE_DIR):
        return
    for fn in os.listdir(CACHE_DIR):
        if dataset is None:
            fp = os.path.join(CACHE_DIR, fn)
            os.remove(fp)
            continue
        # 严格前缀匹配: 避免 "SEED" 误匹配 "SEED_IV"
        # "SEED_IV" 应该匹配 "SEED_IV_" 前缀, "SEED" 应该匹配 "SEED_" 但不匹配 "SEED_IV_"
        if fn.startswith(f"{dataset}_"):
            # 额外检查: 若 dataset="SEED", 排除 "SEED_IV_..."
            if dataset == "SEED" and fn.startswith("SEED_IV_"):
                continue
            fp = os.path.join(CACHE_DIR, fn)
            os.remove(fp)


def cache_info():
    """打印缓存信息."""
    if not os.path.isdir(CACHE_DIR):
        print(f"缓存目录不存在: {CACHE_DIR}")
        return
    files = os.listdir(CACHE_DIR)
    npz_files = [f for f in files if f.endswith(".npz")]
    total_size = sum(
        os.path.getsize(os.path.join(CACHE_DIR, f)) for f in npz_files
    )
    # 按数据集分组统计
    # 注意: 与 clear_cache 同样的前缀匹配问题, "SEED_IV_xxx.npz" 用 split("_")[0] 得到 "SEED",
    # 会被错误归入 SEED 组. 修复: 优先检查 "SEED_IV_" 前缀.
    by_dataset = {}
    for f in npz_files:
        if f.startswith("SEED_IV_"):
            ds = "SEED_IV"
        elif f.startswith("SEED_"):
            ds = "SEED"
        else:
            ds = f.split("_")[0]
        by_dataset.setdefault(ds, 0)
        by_dataset[ds] += 1
    print(f"缓存目录: {CACHE_DIR}")
    print(f"  文件数: {len(npz_files)}")
    print(f"  总大小: {total_size / 1024 / 1024:.1f} MB")
    for ds, count in sorted(by_dataset.items()):
        print(f"    {ds}: {count} 个被试")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="协方差缓存管理")
    p.add_argument("--info", action="store_true", help="显示缓存信息")
    p.add_argument("--clear", action="store_true", help="清空全部缓存")
    p.add_argument("--clear_dataset", type=str, help="清空指定数据集缓存")
    args = p.parse_args()

    if args.info:
        cache_info()
    elif args.clear:
        clear_cache()
        print("已清空全部缓存")
    elif args.clear_dataset:
        clear_cache(args.clear_dataset)
        print(f"已清空 {args.clear_dataset} 缓存")
    else:
        p.print_help()
