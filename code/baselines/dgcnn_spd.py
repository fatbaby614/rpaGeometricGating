"""
baselines/dgcnn_spd.py — DGCNN-SPD: 动态图卷积网络 (SPD 域变体)
==============================================================

经典 DL baseline (Song et al. 2018, IEEE TAFFC, DGCNN) 的 SPD 域变体.

原论文 DGCNN 输入原始 EEG, 学习邻接矩阵 + 图卷积. 本变体直接接收协方差矩阵
(SPD 矩阵), 把协方差矩阵的通道相关性作为图邻接矩阵, 用 GCN 学习通道间关系.

设计选择 (与原论文的差异):
    - 输入: 协方差矩阵 (n_epochs, C, C) 而非原始 EEG (n_epochs, C, T)
    - 图结构: 协方差矩阵本身作为邻接矩阵 (无需学习), GCN 学习节点特征
    - 优势: 与本项目 Riemannian pipeline 共享协方差缓存, 公平对比
    - 局限: 原论文用原始 EEG 可能拿到更高 acc, 本变体是 "SPD-aware" 简化版

参考:
    Song, T., et al. "EEG Emotion Recognition Using Dynamical Graph
    Convolutional Neural Networks." IEEE TAFFC, 2018.

注意:
    - 需要 PyTorch (CPU 或 GPU)
    - LOSO 评测, 每 fold 独立训练 (与 evaluate_loso_classification 一致)
    - 固定 seed, 可复现
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from .base import Baseline, register_baseline, set_global_seed


@register_baseline("DGCNN-SPD")
class DGCNN_SPD(Baseline):
    """DGCNN-SPD: 协方差矩阵作为图邻接矩阵的 GCN.

    网络结构:
        1. 输入: (n_epochs, C, C) 协方差矩阵
        2. 对每行 (每个通道) 做 GCN: H = D^{-1/2} (A+I) D^{-1/2} X W
           其中 A = cov (绝对值, 归一化), X = I (初始节点特征 = 单位阵行)
        3. 两层 GCN + ReLU + Dropout
        4. 全局平均池化 (按通道维) → 全连接 → softmax
    """

    def __init__(self, seed: int = 42, hidden_dim: int = 64,
                 n_epochs: int = 30, lr: float = 1e-3,
                 batch_size: int = 64, dropout: float = 0.3,
                 weight_decay: float = 1e-4, device: str = "auto"):
        super().__init__(seed=seed)
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs      # 训练 epoch 数 (非 EEG epoch)
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.device = device

    def _resolve_device(self):
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _build_model(self, n_channels: int, n_classes: int):
        """构建 DGCNN-SPD 模型."""
        import torch
        import torch.nn as nn

        class DGCNNLayer(nn.Module):
            """单层 GCN: H = D^{-1/2} (A+I) D^{-1/2} X W"""
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.linear = nn.Linear(in_dim, out_dim)

            def forward(self, X, A):
                # A: (B, C, C) 邻接矩阵 (已归一化)
                # X: (B, C, in_dim) 节点特征
                support = self.linear(X)            # (B, C, out_dim)
                out = torch.bmm(A, support)         # (B, C, out_dim)
                return out

        class DGCNNSPDNet(nn.Module):
            def __init__(self, n_channels, hidden_dim, n_classes, dropout):
                super().__init__()
                self.gc1 = DGCNNLayer(n_channels, hidden_dim)
                self.gc2 = DGCNNLayer(hidden_dim, hidden_dim)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim, n_classes)

            def forward(self, cov):
                # cov: (B, C, C) 协方差矩阵
                B, C, _ = cov.shape
                # 构建邻接矩阵: |cov| 归一化 (对称归一化 D^{-1/2} (A+I) D^{-1/2})
                A = torch.abs(cov)
                A = A + torch.eye(C, device=cov.device).unsqueeze(0)  # 加自环
                # 对角矩阵 D
                deg = A.sum(dim=-1, keepdim=True)  # (B, C, 1)
                D_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
                A_norm = A * D_inv_sqrt * D_inv_sqrt.transpose(-1, -2)
                # 初始节点特征: 单位阵 (每个节点 = one-hot)
                X = torch.eye(C, device=cov.device).unsqueeze(0).expand(B, -1, -1)
                # 两层 GCN
                H = torch.relu(self.gc1(X, A_norm))   # (B, C, hidden)
                H = self.dropout(H)
                H = torch.relu(self.gc2(H, A_norm))   # (B, C, hidden)
                H = self.dropout(H)
                # 全局平均池化 (按通道维)
                H = H.mean(dim=1)                     # (B, hidden)
                out = self.fc(H)                      # (B, n_classes)
                return out

        return DGCNNSPDNet(n_channels, self.hidden_dim, n_classes, self.dropout)

    def _normalize_covs(self, covs: np.ndarray) -> np.ndarray:
        """协方差矩阵归一化: trace 归一化到 C (与 RPA reprojection 一致).

        NaN/Inf 防护: 若 covs 含 NaN/Inf (如缓存损坏或 OAS 估计退化),
        trace 会是 NaN, lambdas = C/NaN = NaN, 后续 Linear/bmm 传播 NaN,
        CrossEntropyLoss 产生 NaN loss, Adam 更新后参数全 NaN,
        argmax(NaN) 通常返回 0, 该 fold 静默产出 ~1/n_classes 的错误 acc.
        fold 级 try/except 不会捕获 (无异常抛出), 污染最终 acc_mean.
        修复: 入口检查 NaN/Inf, 抛 ValueError, 由 fold 级 except 捕获并跳过.
        """
        if not np.all(np.isfinite(covs)):
            raise ValueError(
                f"covs 含 NaN/Inf: shape={covs.shape}, "
                f"non-finite count={int(np.sum(~np.isfinite(covs)))}. "
                f"可能原因: 缓存损坏 / OAS 估计退化 / 磁盘 I/O 异常."
            )
        C = covs.shape[-1]
        traces = np.trace(covs, axis1=-2, axis2=-1)  # (n_epochs,)
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

        # 合并频段: 把每个 band 的协方差作为独立通道维 (concatenate 沿特征维)
        # 或: 对每个 epoch, 把 5 个 band 的 (C,C) 平均成单个 (C,C) 协方差
        # 选择: 平均 (简单, 与 SVM baseline 的 FBTS 不同但合理)
        acc_list, f1_list = [], []
        all_y_true, all_y_pred = [], []
        t_total_start = time.time()

        # 确定 n_classes (从所有标签统计)
        all_labels = np.concatenate([all_y[s] for s in subjects])
        n_classes = int(all_labels.max()) + 1
        n_channels = all_covs[subjects[0]][bands[0]].shape[-1]

        print(f"\n{'='*60}")
        print(f"LOSO + DGCNN-SPD (PyTorch, device={device})")
        print(f"  Channels: {n_channels}, Bands: {len(bands)}, Classes: {n_classes}")
        print(f"  hidden_dim={self.hidden_dim}, n_epochs={self.n_epochs}, lr={self.lr}")
        print(f"{'='*60}")

        for i, test_subj in enumerate(subjects):
            t_fold = time.time()
            # ⚠️ fold 级 try/except: 单 fold OOM/CUDA error 不应让整个 baseline 失败
            # 与 _run_itsa_baseline 的单类保护一致, 跳过该 fold, acc_list 不追加
            try:
                train_subjs = [s for s in subjects if s != test_subj]

                # 单类保护: 训练集只有一类时, CrossEntropyLoss 退化为恒预测该类
                train_labels = np.concatenate([all_y[s] for s in train_subjs])
                if len(np.unique(train_labels)) < 2:
                    if verbose:
                        print(f"  [{i+1}/{len(subjects)}] {test_subj}: "
                              f"SKIP (训练集仅含单类)", flush=True)
                    continue

                # 构建训练集 (合并所有 band 的协方差, 取平均)
                X_train, y_train = [], []
                for s in train_subjs:
                    covs_avg = np.mean([all_covs[s][b] for b in bands], axis=0)
                    covs_avg = self._normalize_covs(covs_avg)
                    X_train.append(covs_avg)
                    y_train.append(all_y[s])
                X_train = np.concatenate(X_train, axis=0)  # (N_train, C, C)
                y_train = np.concatenate(y_train, axis=0)

                # 测试集
                covs_test_avg = np.mean([all_covs[test_subj][b] for b in bands], axis=0)
                X_test = self._normalize_covs(covs_test_avg)  # (N_test, C, C)
                y_test = all_y[test_subj]

                # 转 tensor
                X_train_t = torch.FloatTensor(X_train).to(device)
                y_train_t = torch.LongTensor(y_train).to(device)
                X_test_t = torch.FloatTensor(X_test).to(device)
                y_test_t = torch.LongTensor(y_test).to(device)

                train_ds = TensorDataset(X_train_t, y_train_t)
                train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                           shuffle=True, drop_last=False)

                # 构建模型 + 优化器
                model = self._build_model(n_channels, n_classes).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                              weight_decay=self.weight_decay)
                criterion = nn.CrossEntropyLoss()

                # 训练
                model.train()
                for epoch in range(self.n_epochs):
                    for xb, yb in train_loader:
                        optimizer.zero_grad()
                        out = model(xb)
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
                "DGCNN-SPD: 所有 fold 均失败, 无法计算聚合指标. "
                "请检查 GPU/内存/数据完整性."
            )

        t_total = time.time() - t_total_start
        labels_sorted = sorted(set(all_y_true) | set(all_y_pred))
        cm = confusion_matrix(all_y_true, all_y_pred, labels=labels_sorted)

        results = {
            'acc_mean': float(np.mean(acc_list)),
            'acc_std': float(np.std(acc_list)),
            'acc_per_subject': [float(x) for x in acc_list],
            'f1_mean': float(np.mean(f1_list)),
            'f1_std': float(np.std(f1_list)),
            'f1_per_subject': [float(x) for x in f1_list],
            'n_subjects': len(acc_list),  # 实际完成 fold 数 (跳过的 fold 不计入)
            'n_subjects_total': len(subjects),  # 原始被试数 (供 T9 长度校验参考)
            'aligner': 'none',  # DGCNN 不用对齐器
            'bands': f"{len(bands)}band",
            'channels': None,  # DL baseline 不走 channels 配置 (用 covs 原通道数)
            'max_train_epochs_per_subj': None,  # DL baseline 不做 epoch 子采样
            'classifier': 'dgcnn-spd',
            'n_features': n_features,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_labels': [int(x) for x in labels_sorted],
            'time_s': float(t_total),
            'method': 'DGCNN-SPD (Song 2018, SPD variant)',
        }

        print(f"\n  === DGCNN-SPD: ACC={results['acc_mean']:.4f}±"
              f"{results['acc_std']:.4f}, F1={results['f1_mean']:.4f}±"
              f"{results['f1_std']:.4f}")

        return results
