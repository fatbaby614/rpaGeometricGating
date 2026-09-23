"""
baselines/base.py — DL baseline 抽象基类 + 工厂函数
==================================================

所有 DL baseline 继承 Baseline, 实现 evaluate_loso() 方法.
T8 通过 get_baseline(name) 创建实例, 自动处理 GPU/seed/数据接口.

数据接口约定:
    - SPD-based baseline (DGCNN-SPD): 接收协方差矩阵 {sub: {band: (n_epochs, C, C)}}
    - Raw-EEG baseline (mdJPT):      接收原始 EEG {sub: (n_epochs, C, T)}
      若仅有协方差, mdJPT wrapper 会尝试用 covs 近似 (或抛 NotImplementedError)

复现性:
    所有 baseline 在 evaluate_loso() 开头固定 seed (torch + numpy + random).
"""
from __future__ import annotations

import abc
import random
from typing import Dict, List, Tuple

import numpy as np


def set_global_seed(seed: int = 42):
    """固定全局随机种子, 保证 DL baseline 可复现."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cudnn 确定性 (可能牺牲少量性能)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class Baseline(abc.ABC):
    """DL baseline 抽象基类."""

    name: str = "baseline"
    requires_gpu: bool = False
    requires_raw_eeg: bool = False  # True=需原始 EEG; False=SPD 协方差即可

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.kwargs = kwargs

    @abc.abstractmethod
    def evaluate_loso(
        self,
        subjects: List[str],
        all_covs: Dict[str, Dict[Tuple[float, float], np.ndarray]],
        all_y: Dict[str, np.ndarray],
        bands: List[Tuple[float, float]],
        n_features: int = 150,
        verbose: bool = True,
    ) -> Dict:
        """LOSO 评测, 返回与 evaluate_loso_classification 一致的结果格式.

        Args:
            subjects: 被试 ID 列表
            all_covs: {sub: {band: (n_epochs, C, C)}}
            all_y: {sub: (n_epochs,)}
            bands: 频段列表 [(lo, hi), ...]
            n_features: 特征数 (DL baseline 可忽略, 保留接口一致)
            verbose: 是否打印进度

        Returns:
            dict: {acc_mean, acc_std, f1_mean, f1_std, acc_per_subject,
                   f1_per_subject, n_subjects, aligner, bands, classifier,
                   confusion_matrix, confusion_matrix_labels, time_s}
        """
        ...


# ── 工厂函数 ──

_REGISTRY = {}  # name -> class


def register_baseline(name: str):
    """装饰器: 注册 baseline 类."""
    def decorator(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_baseline(name: str, **kwargs) -> Baseline:
    """创建 baseline 实例.

    Args:
        name: baseline 名称 (如 'DGCNN-SPD', 'mdJPT')
        **kwargs: 传给 baseline 构造函数

    Returns:
        Baseline 实例

    Raises:
        KeyError: 未知 baseline
        ImportError: baseline 依赖未安装
    """
    if name not in _REGISTRY:
        # 延迟导入, 避免未安装依赖时 __init__ 报错
        if name in ("DGCNN-SPD", "DGCNN"):
            from .dgcnn_spd import DGCNN_SPD  # noqa: F401
        elif name in ("mdJPT",):
            from .mdjpt_wrapper import mdJPT  # noqa: F401
        else:
            available = list(_REGISTRY.keys())
            raise KeyError(f"未知 baseline: {name}. 可用: {available}")

    cls = _REGISTRY[name]
    return cls(**kwargs)


def list_baselines() -> List[str]:
    """列出所有已注册的 baseline."""
    return list(_REGISTRY.keys())
