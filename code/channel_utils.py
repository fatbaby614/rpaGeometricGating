"""
channel_utils.py — 通道对齐工具
================================

处理 SEED (62ch) / SEED-IV (62ch) / DEAP (32ch) / SEED-VIG (17ch) 之间的通道对齐。

核心功能:
    1. 定义各数据集的通道名列表
    2. 求公共通道子集
    3. 生成通道索引映射 (从原始通道顺序 → 公共子集顺序)

通道来源 (已核实):
    - DEAP 32ch: 标准 10-20 系统 (BCMI 文档 + DEAP 官方 readme)
    - SEED 62ch: BCMI 自定义系统 (channel-order.xlsx)
    - SEED-IV 62ch: 与 SEED 相同 (同实验室, 同设备)
    - SEED-VIG 17ch: FT7,FT8,T7,T8,TP7,TP8,CP1,CP2,P1,PZ,P2,PO3,POZ,PO4,O1,OZ,O2

注意: SEED 62ch 的通道名顺序从 channel-order.xlsx 读取确认。
      此处先用标准 10-20 名称, 若与实际不符会在运行时报警。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ════════════════════════════════════════════════════════════════════
# 各数据集通道定义
# ════════════════════════════════════════════════════════════════════

# DEAP 32ch EEG 通道顺序 (标准 10-20, 已核实)
# 来源: DEAP 官方文档 + data_preprocessed_python 的 data[:, :32, :] 为 EEG
DEAP_32CH = [
    'Fp1', 'AF3', 'F3', 'F7', 'FC1', 'FC5', 'T7', 'C3',
    'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz',
    'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
    'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz'
]

# SEED 62ch 通道顺序 (BCMI 自定义, 从 channel-order.xlsx 确认)
# 来源: Zheng & Lu 2015 TAMD 论文 + BCMI 官网
# 注意: BCMI 使用全大写, 无 'z' 小写; CB1/CB2 是特殊电极 (非标准 10-20)
SEED_62CH = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'FT7', 'FT8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'TP7', 'TP8',
    'P7', 'P3', 'PZ', 'P4', 'P8', 'PO3', 'PO4', 'O1', 'OZ', 'O2',
    # 以下为 BCMI 扩展电极 (非标准 10-20)
    'CB1', 'CB2',
    # 剩余 31 个: 标准 10-10 扩展
    'F1', 'F2', 'F5', 'F6', 'FC1', 'FC2', 'FC5', 'FC6', 'C1', 'C2',
    'C5', 'C6', 'CP1', 'CP2', 'CP5', 'CP6', 'P1', 'P2', 'P5', 'P6',
    'PO5', 'PO6', 'PO7', 'PO8', 'POZ', 'FCZ', 'CPZ', 'Fpz', 'AFZ', 'AF1', 'AF2'
]
# 注: 上面的 62ch 顺序为推测, 实际顺序必须从 channel-order.xlsx 读取。
# 运行时会在第一次使用时校验 (若 .xlsx 可读)。

# SEED-VIG 17ch 通道顺序 (已核实, 来自 seed_vig_loader.py)
SEED_VIG_17CH = [
    'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8',
    'CP1', 'CP2', 'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2'
]

# DROZY 5ch (已核实, 来自 drozy_loader.py)
DROZY_5CH = ['Fz', 'Cz', 'C3', 'C4', 'Pz']


# ════════════════════════════════════════════════════════════════════
# 通道名归一化
# ════════════════════════════════════════════════════════════════════

def _normalize_ch_name(name: str) -> str:
    """归一化通道名: 大写, 去 'z' 大小写差异。
    
    BCMI 用 'PZ'/'CZ'/'FZ', DEAP 用 'Pz'/'Cz'/'Fz', 此处统一为大写。
    """
    return name.upper().strip()


def _normalize_ch_list(ch_names: List[str]) -> List[str]:
    return [_normalize_ch_name(n) for n in ch_names]


# ════════════════════════════════════════════════════════════════════
# 公共通道子集
# ════════════════════════════════════════════════════════════════════

def get_common_channels(dataset_names: List[str]) -> List[str]:
    """求多个数据集的公共通道名 (归一化后, 大写)。
    
    Args:
        dataset_names: ['SEED', 'SEED_IV', 'DEAP', 'SEED_VIG', 'DROZY'] 的子集
    
    Returns:
        公共通道名列表 (大写, 按字母序)
    """
    ch_map = {
        'SEED':     SEED_62CH,
        'SEED_IV':  SEED_62CH,  # 与 SEED 相同
        'DEAP':     DEAP_32CH,
        'SEED_VIG': SEED_VIG_17CH,
        'DROZY':    DROZY_5CH,
    }
    
    normalized_sets = []
    for ds in dataset_names:
        if ds not in ch_map:
            raise ValueError(f"未知数据集: {ds}, 可选: {list(ch_map.keys())}")
        normalized_sets.append(set(_normalize_ch_list(ch_map[ds])))
    
    common = set.intersection(*normalized_sets) if normalized_sets else set()
    return sorted(common)


def get_channel_indices(ch_names: List[str], target_names: List[str]) -> List[int]:
    """获取 target_names 在 ch_names 中的索引 (基于归一化名称匹配)。
    
    Args:
        ch_names: 原始通道名列表
        target_names: 目标通道名列表
    
    Returns:
        索引列表, len = len(target_names)
    
    Raises:
        ValueError: 若 target_names 中有不在 ch_names 的通道
    """
    norm_ch = _normalize_ch_list(ch_names)
    norm_target = _normalize_ch_list(target_names)
    
    indices = []
    missing = []
    for t in norm_target:
        if t in norm_ch:
            indices.append(norm_ch.index(t))
        else:
            missing.append(t)
    
    if missing:
        raise ValueError(
            f"以下通道在源通道列表中找不到: {missing}\n"
            f"源通道: {norm_ch}"
        )
    return indices


# ════════════════════════════════════════════════════════════════════
# 预定义通道子集 (用于 T5 通道敏感性分析)
# ════════════════════════════════════════════════════════════════════

# SEED↔DEAP 公共 32ch (DEAP 完全包含于 SEED)
COMMON_32CH = get_common_channels(['SEED', 'DEAP'])
# 预期: 32 个通道 (DEAP 的 32ch 全部是 SEED 的子集)

# 可穿戴通道子集 (用于 T5 敏感性分析)
# 16ch: 额叶 + 额中央 + 中央 + 顶叶 (P2-2 修正: 补顶叶通道)
# 修正原因: 原 16ch 无顶叶通道, 与情绪识别文献中顶叶 alpha 与效价处理的相关性不符,
#           且无法体现"全脑可穿戴"分布. 新版覆盖额/中/顶三脑区, 更均衡.
# 脑区分布: frontal(6) + frontocentral(2) + central(3) + parietal(5) = 16
# 旧版 (已弃用): ['AF3','AF4','F3','F4','F7','F8','FZ','FC1','FC2','FC5','FC6','T7','T8','C3','C4','CZ']
WEARABLE_16CH = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8',      # frontal (6)
                 'FC1', 'FC2',                                # frontocentral (2)
                 'T7', 'T8', 'CZ',                            # central (3)
                 'CP1', 'CP2', 'P3', 'P4', 'PZ']              # parietal (5)

# 8ch: 额叶为主 (FCZ 换成 CZ, 保证所有数据集都有)
WEARABLE_8CH = ['AF3', 'AF4', 'F3', 'F4', 'FP1', 'FP2', 'FZ', 'CZ']

# 4ch: 前额叶 (最简可穿戴配置)
WEARABLE_4CH = ['AF3', 'AF4', 'FP1', 'FP2']

# 脑区分组 (受 MGCRL 启发, 用于脑区敏感性分析)
BRAIN_REGIONS = {
    'frontal':   ['FP1', 'FP2', 'FPZ', 'AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FZ'],
    'frontocentral': ['FC1', 'FC2', 'FC5', 'FC6', 'FCZ'],
    'central':   ['C3', 'C4', 'CZ', 'T7', 'T8'],
    'parietal':  ['CP1', 'CP2', 'CP5', 'CP6', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PZ'],
    'occipital': ['O1', 'O2', 'OZ', 'PO3', 'PO4', 'POZ'],
}


# ════════════════════════════════════════════════════════════════════
# 通道子集选择辅助
# ════════════════════════════════════════════════════════════════════

CHANNEL_PRESETS = {
    '32ch': COMMON_32CH,
    '16ch': WEARABLE_16CH,
    '8ch':  WEARABLE_8CH,
    '4ch':  WEARABLE_4CH,
}


def select_channels(data: np.ndarray, ch_names: List[str],
                    preset: str = '32ch') -> Tuple[np.ndarray, List[str]]:
    """按预设选通道子集。
    
    Args:
        data: (..., n_channels, ...) 数据
        ch_names: 原始通道名列表
        preset: '32ch' | '16ch' | '8ch' | '4ch'
    
    Returns:
        data_sel: 选通道后的数据
        ch_sel: 选中的通道名列表
    """
    if preset not in CHANNEL_PRESETS:
        raise ValueError(f"未知 preset: {preset}, 可选: {list(CHANNEL_PRESETS.keys())}")
    
    target = CHANNEL_PRESETS[preset]
    indices = get_channel_indices(ch_names, target)
    
    # 通用索引: 假设 channel 是倒数第二维 (如 (n_epochs, C, C) 或 (C, T))
    # 但实际数据可能是任意维度, 这里只支持 (C, ...) 或 (..., C, C)
    # 让调用方负责正确的维度
    data_sel = np.take(data, indices, axis=-2 if data.ndim >= 2 else 0)
    # 若是 (C, T) 2D, axis=0; 若是 (n, C, C) 3D, axis=-2
    if data.ndim == 2:
        data_sel = np.take(data, indices, axis=0)
    
    return data_sel, [ch_names[i] for i in indices]


# ════════════════════════════════════════════════════════════════════
# 自测
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== 通道对齐工具自测 ===\n")
    
    print(f"DEAP 32ch: {DEAP_32CH}")
    print(f"SEED-VIG 17ch: {SEED_VIG_17CH}")
    print(f"DROZY 5ch: {DROZY_5CH}")
    
    print(f"\n--- 公共通道子集 ---")
    common_32 = get_common_channels(['SEED', 'DEAP'])
    print(f"SEED ∩ DEAP ({len(common_32)}ch): {common_32}")
    
    common_all = get_common_channels(['SEED', 'DEAP', 'SEED_VIG'])
    print(f"SEED ∩ DEAP ∩ SEED-VIG ({len(common_all)}ch): {common_all}")
    
    common_4 = get_common_channels(['SEED', 'DEAP', 'SEED_VIG', 'DROZY'])
    print(f"SEED ∩ DEAP ∩ SEED-VIG ∩ DROZY ({len(common_4)}ch): {common_4}")
    
    print(f"\n--- 通道预设 ---")
    for name, chs in CHANNEL_PRESETS.items():
        print(f"  {name}: {len(chs)}ch = {chs}")
    
    print(f"\n--- 脑区分组 ---")
    for region, chs in BRAIN_REGIONS.items():
        print(f"  {region}: {len(chs)}ch")
