"""
paths.py — 数据集路径自动解析（跨平台 Windows/Linux）
====================================================

优先级:
    1. 代码内常量 DATA_ROOT (本文件顶部, 最推荐 — 免环境变量)
       - 例:  DATA_ROOT = "/mnt/data1/home/tanhuang/datasets"
       - 会在该目录下查找 SEED / SEED_IV / DEAP / DROZY 子文件夹
    2. 环境变量 RIEMANN_EMO_DATA_ROOT (Linux 工作站备选)
       - 例:  export RIEMANN_EMO_DATA_ROOT=/mnt/data1/home/tanhuang/datasets
    3. 单数据集环境变量 (SEED_ROOT / SEED_IV_ROOT / DEAP_ROOT / DROZY_ROOT)
    4. 操作系统默认路径:
       - Windows: E:\\datasets\\emotion\\{SEED,SEED_IV,DEAP}, E:\\datasets\\DROZY
       - Linux:   /mnt/data1/home/tanhuang/datasets/{SEED,SEED_IV,DEAP,DROZY}

使用:
    from paths import SEED_ROOT, SEED_IV_ROOT, DEAP_ROOT, DROZY_ROOT

切换机器时只需改下面 DATA_ROOT 一行, 无需设环境变量.
"""
from __future__ import annotations

import os
import platform
from typing import List, Optional


# ════════════════════════════════════════════════════════════════════
# 用户配置区: 数据根目录 (留空则用环境变量和 OS 默认路径自动解析)
# ────────────────────────────────────────────────────────────────────
# 切换机器时, 只需改这一行:
#   Windows:  DATA_ROOT = r"E:\datasets\emotion"
#   Linux:    DATA_ROOT = "/mnt/data1/home/tanhuang/datasets"
# 留空 "" 则回退到环境变量 RIEMANN_EMO_DATA_ROOT 和 OS 默认路径.
#
# 注意: SEED 实际数据在 {DATA_ROOT}/SEED/SEED/ (BCMI 多了一层),
#       下面 LINUX_CANDIDATES / WINDOWS_CANDIDATES 已处理此细节.
#       若用 DATA_ROOT, 解析逻辑会尝试 {DATA_ROOT}/{name} 和 {DATA_ROOT}/{name}/{name}.
DATA_ROOT = ""

# 特殊子目录覆盖 (可选, 留空用默认)
# 例: SEED 数据在 /mnt/data1/home/tanhuang/datasets/SEED/SEED,
#     若你的目录结构不同, 可在此显式指定.
DATASET_SUBPATHS = {
    # "SEED":    "SEED/SEED",      # BCMI 多一层
    # "SEED_IV": "SEED_IV",
    # "DEAP":    "DEAP",
    # "DROZY":   "DROZY",
}


def _is_dir(p: Optional[str]) -> bool:
    return bool(p) and os.path.isdir(p)


def _join(base: str, sub: str) -> str:
    return os.path.join(base, sub)


# ───────────────────────────────────────────────────────────────────
# 已知路径候选 (按优先级)
# ───────────────────────────────────────────────────────────────────
WINDOWS_CANDIDATES = {
    "SEED":    [r"E:\datasets\emotion\SEED\SEED", r"E:\datasets\SEED"],
    "SEED_IV": [r"E:\datasets\emotion\SEED_IV", r"E:\datasets\SEED_IV"],
    "DEAP":    [r"E:\datasets\emotion\DEAP", r"E:\datasets\DEAP"],
    "DROZY":   [r"E:\datasets\DROZY"],
    "SEED-VIG":[r"E:\datasets\SEED-VIG"],
}

LINUX_CANDIDATES = {
    "SEED":    ["/mnt/data1/home/tanhuang/datasets/SEED"],
    "SEED_IV": ["/mnt/data1/home/tanhuang/datasets/SEED_IV"],
    "DEAP":    ["/mnt/data1/home/tanhuang/datasets/DEAP"],
    "DROZY":   ["/mnt/data1/home/tanhuang/datasets/DROZY"],
    "SEED-VIG":["/mnt/data1/home/tanhuang/datasets/SEED-VIG"],
}


def _resolve(dataset: str) -> str:
    """解析数据集路径, 返回实际存在的目录. 找不到时 raise."""
    candidates: List[str] = []

    # 0) 代码内常量 DATA_ROOT (最优先, 免环境变量)
    if DATA_ROOT:
        # 用户显式覆盖子路径 (DATASET_SUBPATHS)
        if dataset in DATASET_SUBPATHS:
            candidates.append(_join(DATA_ROOT, DATASET_SUBPATHS[dataset]))
        else:
            # 默认尝试两种结构: {ROOT}/{name} 和 {ROOT}/{name}/{name} (BCMI SEED/SEED)
            candidates.append(_join(DATA_ROOT, dataset))
            candidates.append(_join(_join(DATA_ROOT, dataset), dataset))

    # 1) 单数据集环境变量 (SEED_ROOT, SEED_IV_ROOT, DEAP_ROOT, DROZY_ROOT, SEED_VIG_ROOT)
    env_key = dataset.replace("-", "_")
    env_specific = os.environ.get(f"{env_key}_ROOT")
    if env_specific:
        candidates.append(env_specific)

    # 2) 通用数据根环境变量 (Linux 工作站备选)
    env_root = os.environ.get("RIEMANN_EMO_DATA_ROOT")
    if env_root:
        candidates.append(_join(env_root, dataset))

    # 3) 操作系统默认
    sysname = platform.system().lower()
    if "windows" in sysname:
        candidates.extend(WINDOWS_CANDIDATES.get(dataset, []))
    else:  # linux / darwin
        candidates.extend(LINUX_CANDIDATES.get(dataset, []))

    for c in candidates:
        if _is_dir(c):
            return c

    tried = "\n  ".join(candidates) if candidates else "(no candidates)"
    raise FileNotFoundError(
        f"找不到 {dataset} 数据集. 已尝试:\n  {tried}\n"
        f"可在 paths.py 顶部设置 DATA_ROOT, 或用环境变量 "
        f"RIEMANN_EMO_DATA_ROOT / {env_key}_ROOT 指定."
    )


# ───────────────────────────────────────────────────────────────────
# 公共 API
# ───────────────────────────────────────────────────────────────────
_cache: dict = {}


def get_path(dataset: str) -> str:
    """获取数据集路径, 失败时给出明确错误信息."""
    if dataset not in _cache:
        _cache[dataset] = _resolve(dataset)
    return _cache[dataset]


def __getattr__(name: str) -> str:
    """懒加载: SEED_ROOT / SEED_IV_ROOT / DEAP_ROOT / DROZY_ROOT / SEED_VIG_ROOT"""
    mapping = {
        "SEED_ROOT":    "SEED",
        "SEED_IV_ROOT": "SEED_IV",
        "DEAP_ROOT":    "DEAP",
        "DROZY_ROOT":   "DROZY",
        "SEED_VIG_ROOT":"SEED-VIG",
    }
    if name in mapping:
        return get_path(mapping[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 显式常量 (import 时解析, 让失败更早暴露)
# 用 try/except 避免某个数据集缺失时阻塞其他数据集
try:
    SEED_ROOT = get_path("SEED")
except FileNotFoundError:
    SEED_ROOT = None

try:
    SEED_IV_ROOT = get_path("SEED_IV")
except FileNotFoundError:
    SEED_IV_ROOT = None

try:
    DEAP_ROOT = get_path("DEAP")
except FileNotFoundError:
    DEAP_ROOT = None

try:
    DROZY_ROOT = get_path("DROZY")
except FileNotFoundError:
    DROZY_ROOT = None

try:
    SEED_VIG_ROOT = get_path("SEED-VIG")
except FileNotFoundError:
    SEED_VIG_ROOT = None


if __name__ == "__main__":
    print(f"Platform: {platform.system()}")
    print(f"  SEED_ROOT:     {SEED_ROOT}")
    print(f"  SEED_IV_ROOT:  {SEED_IV_ROOT}")
    print(f"  DEAP_ROOT:     {DEAP_ROOT}")
    print(f"  DROZY_ROOT:    {DROZY_ROOT}")
    print(f"  SEED_VIG_ROOT: {SEED_VIG_ROOT}")
    print(f"  RIEMANN_EMO_DATA_ROOT = {os.environ.get('RIEMANN_EMO_DATA_ROOT', '(not set)')}")
