"""
baselines/mdjpt_wrapper.py — mdJPT baseline wrapper
==================================================

最新 DL baseline (Liu et al. 2025, NeurIPS, arXiv:2510.22197):
    "masked Joint Pre-Training for EEG Emotion Recognition"
    使用 cross-dataset covariance alignment loss, 与本项目跨库迁移实验对应.

两种模式:
    1. 原仓库模式 (优先): 若用户已 clone mdJPT 仓库到 MDJPT_ROOT 环境变量指向的路径,
       则 import 原始模型, 用本项目数据接口适配.
    2. Fallback 模式: 若原仓库未安装, 用本项目自实现的轻量级 SPD-Transformer
       (协方差矩阵 + Transformer encoder) 作为近似 baseline.
       ⚠️ Fallback 与原 mdJPT 不等价, 仅用于"代码可运行"验证, 论文数值需用原仓库.

环境变量:
    MDJPT_ROOT: mdJPT 仓库根目录 (含 model.py 或 mdjpt/ 子目录)
        例: export MDJPT_ROOT=/home/user/mdjpt

参考:
    Liu, Q., et al. "Masked Joint Pre-Training for Cross-Dataset EEG
    Emotion Recognition." NeurIPS, 2025. arXiv:2510.22197
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import numpy as np

from .base import Baseline, register_baseline, set_global_seed


def _try_import_original_mdjpt():
    """尝试 import 原始 mdJPT 模型.

    Returns:
        (model_factory, mode) 或 (None, None)
        model_factory: 接收 (n_channels, n_classes, **kwargs) 返回 model 实例
        mode: 'original' | 'fallback'
    """
    mdjpt_root = os.environ.get("MDJPT_ROOT", "")
    if not mdjpt_root or not os.path.isdir(mdjpt_root):
        return None, None

    # 尝试把 mdjpt_root 加入 sys.path
    import sys
    if mdjpt_root not in sys.path:
        sys.path.insert(0, mdjpt_root)

    # 尝试不同的 import 路径 (原仓库可能有不同结构)
    try:
        # 尝试 1: from mdjpt.model import mdJPTModel
        try:
            from mdjpt.model import mdJPTModel  # type: ignore
            return (lambda n_ch, n_cls, **kw: mdJPTModel(
                n_channels=n_ch, n_classes=n_cls, **kw)), 'original'
        except ImportError:
            pass

        # 尝试 2: from model import mdJPTModel
        try:
            from model import mdJPTModel  # type: ignore
            return (lambda n_ch, n_cls, **kw: mdJPTModel(
                n_channels=n_ch, n_classes=n_cls, **kw)), 'original'
        except ImportError:
            pass

        # 尝试 3: 直接 import mdjpt
        try:
            import mdjpt  # type: ignore
            if hasattr(mdjpt, 'mdJPTModel'):
                return (lambda n_ch, n_cls, **kw: mdjpt.mdJPTModel(
                    n_channels=n_ch, n_classes=n_cls, **kw)), 'original'
        except ImportError:
            pass

    except Exception as e:
        print(f"[mdJPT] import 原仓库失败: {e}, 使用 fallback")

    return None, None


@register_baseline("mdJPT")
class mdJPT(Baseline):
    """mdJPT baseline (Liu 2025 NeurIPS) wrapper.

    优先用原仓库; 若未安装, 用 SPD-Transformer fallback (自实现).
    """

    def __init__(self, seed: int = 42, hidden_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 2,
                 n_epochs: int = 30, lr: float = 5e-4,
                 batch_size: int = 32, dropout: float = 0.1,
                 weight_decay: float = 1e-4, device: str = "auto",
                 mask_ratio: float = 0.15):
        super().__init__(seed=seed)
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.device = device
        self.mask_ratio = mask_ratio  # mdJPT 的 masked pretraining 比例

        # 尝试加载原仓库
        self._orig_factory, self._mode = _try_import_original_mdjpt()
        if self._mode == 'original':
            print(f"[mdJPT] 使用原仓库模式 (MDJPT_ROOT={os.environ.get('MDJPT_ROOT')})")
        else:
            print(f"[mdJPT] 原仓库未安装, 使用 SPD-Transformer fallback")
            print(f"  ⚠️ Fallback 与原 mdJPT 不等价, 论文数值需用原仓库")
            print(f"  安装: git clone https://arxiv.org/abs/2510.22197 → 设置 MDJPT_ROOT")

    def _resolve_device(self):
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _build_fallback_model(self, n_channels: int, n_classes: int):
        """Fallback: SPD-Transformer (自实现).

        结构:
            1. 输入: (B, C, C) 协方差矩阵
            2. 对每行 (通道) 做线性投影到 hidden_dim → (B, C, hidden_dim)
            3. Transformer encoder (n_layers 层, n_heads 头)
            4. 全局平均池化 → 全连接 → softmax
        """
        import torch
        import torch.nn as nn

        class SPDTransformer(nn.Module):
            def __init__(self, n_channels, hidden_dim, n_classes,
                         n_heads, n_layers, dropout):
                super().__init__()
                self.proj = nn.Linear(n_channels, hidden_dim)
                # Transformer encoder: batch_first=True
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=n_heads,
                    dim_feedforward=hidden_dim * 4, dropout=dropout,
                    batch_first=True)
                self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim, n_classes)

            def forward(self, cov):
                # cov: (B, C, C)
                # 每行 (通道) 作为 token, 投影到 hidden_dim
                H = self.proj(cov)              # (B, C, hidden_dim)
                H = self.encoder(H)             # (B, C, hidden_dim)
                H = H.mean(dim=1)               # (B, hidden_dim) 全局平均池化
                H = self.dropout(H)
                out = self.fc(H)                # (B, n_classes)
                return out

        return SPDTransformer(n_channels, self.hidden_dim, n_classes,
                               self.n_heads, self.n_layers, self.dropout)

    def _build_model(self, n_channels: int, n_classes: int):
        """构建模型 (原仓库或 fallback)."""
        if self._mode == 'original' and self._orig_factory is not None:
            return self._orig_factory(n_channels, n_classes)
        return self._build_fallback_model(n_channels, n_classes)

    def _normalize_covs(self, covs: np.ndarray) -> np.ndarray:
        """协方差矩阵 trace 归一化 (与 DGCNN-SPD 一致).

        NaN/Inf 防护: 与 DGCNN-SPD._normalize_covs 同理, 见该函数注释.
        """
        if not np.all(np.isfinite(covs)):
            raise ValueError(
                f"covs 含 NaN/Inf: shape={covs.shape}, "
                f"non-finite count={int(np.sum(~np.isfinite(covs)))}. "
                f"可能原因: 缓存损坏 / OAS 估计退化 / 磁盘 I/O 异常."
            )
        C = covs.shape[-1]
        traces = np.trace(covs, axis1=-2, axis2=-1)
        lambdas = C / np.maximum(traces, 1e-10)
        return covs * lambdas[:, None, None]

    def evaluate_loso(
        self,
        subjects: List[str],
        all_covs: Dict[str, Dict[Tuple[float, float], np.ndarray]],
        all_y: Dict[str, np.ndarray],
        bands: List[Tuple[float, float]],
        n_features: int = 150,
        verbose: bool = True,
    ) -> Dict:
        """LOSO 评测."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

        set_global_seed(self.seed)
        device = torch.device(self._resolve_device())

        acc_list, f1_list = [], []
        all_y_true, all_y_pred = [], []
        t_total_start = time.time()

        all_labels = np.concatenate([all_y[s] for s in subjects])
        n_classes = int(all_labels.max()) + 1
        n_channels = all_covs[subjects[0]][bands[0]].shape[-1]

        print(f"\n{'='*60}")
        print(f"LOSO + mdJPT (mode={self._mode}, device={device})")
        print(f"  Channels: {n_channels}, Bands: {len(bands)}, Classes: {n_classes}")
        print(f"  hidden_dim={self.hidden_dim}, n_heads={self.n_heads}, "
              f"n_layers={self.n_layers}")
        print(f"  n_epochs={self.n_epochs}, lr={self.lr}, mask_ratio={self.mask_ratio}")
        print(f"{'='*60}")

        for i, test_subj in enumerate(subjects):
            t_fold = time.time()
            # ⚠️ fold 级 try/except: 单 fold OOM/CUDA error 不应让整个 baseline 失败
            # 与 DGCNN-SPD / _run_itsa_baseline 的 fold 容错策略一致
            try:
                train_subjs = [s for s in subjects if s != test_subj]

                # 单类保护: 训练集只有一类时, CrossEntropyLoss 退化为恒预测该类
                train_labels = np.concatenate([all_y[s] for s in train_subjs])
                if len(np.unique(train_labels)) < 2:
                    if verbose:
                        print(f"  [{i+1}/{len(subjects)}] {test_subj}: "
                              f"SKIP (训练集仅含单类)", flush=True)
                    continue

                # 构建训练集 (band 平均, 同 DGCNN-SPD)
                X_train, y_train = [], []
                for s in train_subjs:
                    covs_avg = np.mean([all_covs[s][b] for b in bands], axis=0)
                    covs_avg = self._normalize_covs(covs_avg)
                    X_train.append(covs_avg)
                    y_train.append(all_y[s])
                X_train = np.concatenate(X_train, axis=0)
                y_train = np.concatenate(y_train, axis=0)

                X_test = self._normalize_covs(
                    np.mean([all_covs[test_subj][b] for b in bands], axis=0))
                y_test = all_y[test_subj]

                X_train_t = torch.FloatTensor(X_train).to(device)
                y_train_t = torch.LongTensor(y_train).to(device)
                X_test_t = torch.FloatTensor(X_test).to(device)
                y_test_t = torch.LongTensor(y_test).to(device)

                train_ds = TensorDataset(X_train_t, y_train_t)
                train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                           shuffle=True, drop_last=False)

                model = self._build_model(n_channels, n_classes).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                               weight_decay=self.weight_decay)
                criterion = nn.CrossEntropyLoss()

                # 训练 (带可选的 masked pretraining)
                model.train()
                for epoch in range(self.n_epochs):
                    for xb, yb in train_loader:
                        # mdJPT 风格: 随机 mask 部分 token (通道), 再分类
                        if self.mask_ratio > 0 and self._mode != 'original':
                            B, C, _ = xb.shape
                            mask = torch.rand(B, C, device=xb.device) > self.mask_ratio
                            # 对称 mask: 同时 mask 行和列, 保持协方差矩阵对称性
                            # 否则 xb * mask.unsqueeze(-1) 只 mask 行不 mask 列,
                            # 会导致 xb_masked[i,j] != xb_masked[j,i], 破坏 SPD 性质
                            mask_sym = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float()
                            xb_masked = xb * mask_sym
                            out = model(xb_masked)
                        else:
                            out = model(xb)
                        optimizer.zero_grad()
                        loss = criterion(out, yb)
                        loss.backward()
                        optimizer.step()

                # 测试
                model.eval()
                with torch.no_grad():
                    logits = model(X_test_t)
                    y_pred = logits.argmax(dim=-1).cpu().numpy()

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                acc_list.append(acc)
                f1_list.append(f1)
                all_y_true.extend(list(y_test))
                all_y_pred.extend(list(y_pred))

                if verbose:
                    print(f"  [{i+1}/{len(subjects)}] {test_subj}: "
                          f"ACC={acc:.4f}, F1={f1:.4f} ({time.time()-t_fold:.1f}s)")
            except (RuntimeError, MemoryError, ValueError) as e:
                # GPU OOM / CUDA error / NaN 数据 (ValueError 由 _normalize_covs 抛出):
                # 跳过此 fold, 不阻塞整体
                print(f"  [{i+1}/{len(subjects)}] {test_subj}: "
                      f"SKIP (fold 失败: {type(e).__name__}: {e})", flush=True)
                continue

        # 全部 fold 失败的兜底: 避免下游 np.mean([]) 报错
        if len(acc_list) == 0:
            raise RuntimeError(
                "mdJPT: 所有 fold 均失败, 无法计算聚合指标. "
                "请检查 GPU/内存/数据完整性."
            )

        t_total = time.time() - t_total_start
        labels_sorted = sorted(set(all_y_true) | set(all_y_pred))
        cm = confusion_matrix(all_y_true, all_y_pred, labels=labels_sorted)

        method_note = ('mdJPT (Liu 2025 NeurIPS, original)' if self._mode == 'original'
                       else 'mdJPT-fallback (SPD-Transformer, 自实现, 非原论文)')

        results = {
            'acc_mean': float(np.mean(acc_list)),
            'acc_std': float(np.std(acc_list)),
            'acc_per_subject': [float(x) for x in acc_list],
            'f1_mean': float(np.mean(f1_list)),
            'f1_std': float(np.std(f1_list)),
            'f1_per_subject': [float(x) for x in f1_list],
            'n_subjects': len(acc_list),  # 实际完成 fold 数 (跳过的 fold 不计入)
            'n_subjects_total': len(subjects),  # 原始被试数 (供 T9 长度校验参考)
            'aligner': 'none',
            'bands': f"{len(bands)}band",
            'channels': None,  # DL baseline 不走 channels 配置 (用 covs 原通道数)
            'max_train_epochs_per_subj': None,  # DL baseline 不做 epoch 子采样
            'classifier': 'mdjpt' if self._mode == 'original' else 'mdjpt-fallback',
            'n_features': n_features,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_labels': [int(x) for x in labels_sorted],
            'time_s': float(t_total),
            'method': method_note,
            'mode': self._mode,
        }

        print(f"\n  === mdJPT ({self._mode}): ACC={results['acc_mean']:.4f}±"
              f"{results['acc_std']:.4f}, F1={results['f1_mean']:.4f}±"
              f"{results['f1_std']:.4f}")

        return results
