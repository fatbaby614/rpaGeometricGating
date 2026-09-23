"""
baselines/__init__.py — DL baseline 工厂
=======================================

统一入口: 通过 get_baseline(name) 创建 baseline 实例.

支持的 baselines:
    - DGCNN-SPD: 动态图卷积网络 (SPD 域变体), Song 2018 IEEE TAFFC
    - mdJPT:     masked Joint Pre-Training, Liu 2025 NeurIPS (wrapper)

注意:
    mdJPT 依赖外部仓库 (arXiv:2510.22197), 若未安装则 import 时抛 ImportError.
    DGCNN-SPD 为本项目自实现, 无外部依赖 (除 PyTorch).
"""
from __future__ import annotations

from .base import Baseline, get_baseline, list_baselines

__all__ = ["Baseline", "get_baseline", "list_baselines"]
