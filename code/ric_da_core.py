"""
ric_da_core.py — 核心对齐算法 + 跨被试评测引擎（分类版）
=========================================================

复用自 RiemannianDomainAdaptation/code/ric_da_core.py 的对齐器类，
新增分类评测引擎（原版是回归）。

对齐器（与原版完全一致）:
    1. NoAlignment       — 基线
    2. EuclideanAlignment — 欧氏白化
    3. RiemannianAlignment — 黎曼平行传输 (Zanini 2018)
    4. RiemannianProcrustesAlignment (RPA) — Rodrigues 2019

评测引擎:
    evaluate_loso_classification — 分类 LOSO (Acc/F1)
    evaluate_loso_regression     — 回归 LOSO (Cor/RMSE, 复用原版)
    evaluate_cross_dataset       — 跨数据集迁移 (源域训练→目标域测试)

参考文献:
    [1] Zanini et al., IEEE TBME, 2018.
    [2] Rodrigues et al., IEEE TBME, 2019.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann, mean_euclid
from pyriemann.utils.base import sqrtm, invsqrtm
from pyriemann.utils.distance import distance_riemann

# 频段
BANDS_5 = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]
# 8 频段细分 (标准分解): δ / θ / low-α / high-α / low-β / high-β / low-γ / high-γ
BANDS_8 = [(1, 4), (4, 8), (8, 11), (11, 14),
           (14, 20), (20, 31), (31, 40), (40, 50)]


# ════════════════════════════════════════════════════════════════
# 对齐器基类 + 四种实现 (复用自 RiemannianDomainAdaptation)
# ════════════════════════════════════════════════════════════════

class BaseAlignment:
    """对齐器基类。

    缓存机制 (方案 1 优化):
        `fit`/`align_subject` 支持可选 `band`/`subject` 参数, 用于跨 fold 复用
        被试级黎曼均值. 同一被试同一 band 的 mean_riemann 只算一次, 跨 fold 缓存.
        不传这两个参数时走原逻辑 (每次重算), 保持向后兼容.
    """
    name = 'base'

    def fit(self, covs_per_subject, band=None):
        raise NotImplementedError

    def transform(self, covs, subject_label=None):
        raise NotImplementedError

    def fit_transform(self, covs_per_subject, band=None):
        self.fit(covs_per_subject, band=band)
        return {s: self.transform(c, s) for s, c in covs_per_subject.items()}

    def align_subject(self, covs, subject=None, band=None):
        raise NotImplementedError


class NoAlignment(BaseAlignment):
    name = 'none'
    def fit(self, covs_per_subject, band=None): pass
    def transform(self, covs, subject_label=None): return covs
    def align_subject(self, covs, subject=None, band=None): return covs


class EuclideanAlignment(BaseAlignment):
    """欧氏空间白化对齐 (He & Wu 2018)."""
    name = 'euclidean'

    def __init__(self, reference='identity'):
        self.reference = reference
        self.transforms_ = {}

    def fit(self, covs_per_subject, band=None):
        self.transforms_ = {}
        all_covs_list = []
        for subject, covs in covs_per_subject.items():
            eucl_mean = np.mean(covs, axis=0)
            eucl_mean = (eucl_mean + eucl_mean.T) / 2
            self.transforms_[subject] = invsqrtm(eucl_mean)
            all_covs_list.append(covs)
        if self.reference == 'grand_mean':
            all_covs = np.concatenate(all_covs_list, axis=0)
            grand_mean = np.mean(all_covs, axis=0)
            grand_mean = (grand_mean + grand_mean.T) / 2
            self.grand_sqrt_ = sqrtm(grand_mean)
        else:
            self.grand_sqrt_ = np.eye(covs.shape[-1])

    def transform(self, covs, subject_label=None):
        if subject_label is not None and subject_label in self.transforms_:
            T = self.transforms_[subject_label]
        else:
            G = self.grand_sqrt_
            return G @ covs @ G
        covs_id = T @ covs @ T
        if self.reference == 'grand_mean':
            G = self.grand_sqrt_
            return G @ covs_id @ G
        return covs_id

    def align_subject(self, covs, subject=None, band=None):
        eucl_mean = np.mean(covs, axis=0)
        eucl_mean = (eucl_mean + eucl_mean.T) / 2
        T = invsqrtm(eucl_mean)
        covs_id = T @ covs @ T
        if self.reference == 'grand_mean' and hasattr(self, 'grand_sqrt_'):
            G = self.grand_sqrt_
            return G @ covs_id @ G
        return covs_id


class RiemannianAlignment(BaseAlignment):
    """黎曼流形平行传输对齐 (Zanini et al. 2018).

    对齐步骤 (reference='identity' 默认):
        1. 计算被试 SPD 协方差的黎曼均值 M_sub = mean_riemann(covs_sub)
        2. 平行传输到单位阵: T = M_sub^{-1/2}, cov_aligned = T @ cov @ T
        3. 对齐后所有被试的均值都映射到 I (单位阵), 实现跨被试中心化

    无 reprojection (与 RPA 的关键区别):
        Zanini 2018 仅做平行传输, 不对对齐后矩阵的 trace/determinant 做归一化.
        对齐后协方差的 trace 可能因被试而异 (能量量级未统一).

    reference='grand_mean' 模式 (本实验未用):
        额外乘以 grand_sqrt = sqrtm(mean_riemann(all_covs)), 把所有被试映射到
        全局均值而非单位阵. 适合需要保留全局能量量级的场景.

    参考:
        Zanini, R. A., et al. "Transfer Learning for Brain-Computer Interfaces
        via Riemannian Alignment." IEEE TBME, 2018.
    """
    name = 'riemann'

    def __init__(self, reference='identity', max_iter=50, tol=1e-6):
        self.reference = reference
        self.max_iter = max_iter
        self.tol = tol
        self.transforms_ = {}
        # 被试级黎曼均值缓存: {(subject, band): riem_mean}
        # 跨 fold 复用, mean_riemann 只算一次 (方案 1 优化)
        self._subj_mean_cache = {}

    def _compute_subj_mean(self, covs, subject, band):
        """计算被试级黎曼均值, 带跨 fold 缓存.

        Args:
            covs: (n_epochs, C, C) 被试协方差
            subject: 被试 ID (用于缓存 key)
            band: 频段 (lo, hi), 用于缓存 key

        Returns:
            riem_mean: (C, C) 黎曼均值

        缓存逻辑:
            - 若 subject 和 band 都提供, 且 key 已存在, 直接返回缓存值
            - 否则重新计算 (不缓存或首次计算)
            - 数学等价性: mean_riemann(covs_sub) 只依赖 covs_sub 本身,
              与 fold 无关, 跨 fold 复用完全等价
        """
        cache_key = (subject, band)
        if subject is not None and band is not None and cache_key in self._subj_mean_cache:
            return self._subj_mean_cache[cache_key]
        riem_mean = mean_riemann(covs, maxiter=self.max_iter, tol=self.tol)
        # 对称化: mean_riemann 迭代过程可能引入微小数值不对称, invsqrtm 对此敏感
        riem_mean = (riem_mean + riem_mean.T) / 2
        if subject is not None and band is not None:
            self._subj_mean_cache[cache_key] = riem_mean
        return riem_mean

    def fit(self, covs_per_subject, band=None):
        self.transforms_ = {}
        all_covs_list = []
        for subject, covs in covs_per_subject.items():
            riem_mean = self._compute_subj_mean(covs, subject, band)
            self.transforms_[subject] = invsqrtm(riem_mean)
            all_covs_list.append(covs)
        if self.reference == 'grand_mean':
            all_covs = np.concatenate(all_covs_list, axis=0)
            grand_riem = mean_riemann(all_covs, maxiter=self.max_iter, tol=self.tol)
            grand_riem = (grand_riem + grand_riem.T) / 2  # 对称化
            self.grand_sqrt_ = sqrtm(grand_riem)
        else:
            self.grand_sqrt_ = np.eye(covs.shape[-1])

    def transform(self, covs, subject_label=None):
        if subject_label is not None and subject_label in self.transforms_:
            T = self.transforms_[subject_label]
        else:
            G = self.grand_sqrt_
            return G @ covs @ G
        covs_id = T @ covs @ T
        if self.reference == 'grand_mean':
            G = self.grand_sqrt_
            return G @ covs_id @ G
        return covs_id

    def align_subject(self, covs, subject=None, band=None):
        riem_mean = self._compute_subj_mean(covs, subject, band)
        T = invsqrtm(riem_mean)
        covs_id = T @ covs @ T
        if self.reference == 'grand_mean' and hasattr(self, 'grand_sqrt_'):
            G = self.grand_sqrt_
            return G @ covs_id @ G
        return covs_id


class RiemannianProcrustesAlignment(BaseAlignment):
    """Riemannian Procrustes Analysis (Rodrigues et al. 2019).

    对齐步骤 (reference='identity' 默认):
        1. 计算被试 SPD 协方差的黎曼均值 M_sub = mean_riemann(covs_sub)
        2. 平行传输到单位阵: T = M_sub^{-1/2}, cov_id = T @ cov @ T
           (此步与 RiemannianAlignment 相同)
        3. **Reprojection (RPA 独有)**: trace 归一化
              d = cov.shape[-1]  (通道数)
              λ_i = d / trace(cov_id_i)
              cov_aligned_i = λ_i * cov_id_i
           归一化后所有 cov 的 trace = d, 即协方差矩阵的"总能量"被统一.
           这一步对应 Rodrigues 2019 论文中的 "Procrustes" 部分:
           在 SPD 流形上做 Procrustes 匹配, 把对齐后矩阵投影到
           {X ∈ SPD : trace(X) = d} 子流形上.

    Reprojection 的作用 (与 RiemannianAlignment 的关键区别):
        - RiemannianAlignment 仅做平行传输, 不统一能量量级
        - RPA 额外做 trace 归一化, 消除被试间整体能量差异
        - 在跨数据集场景下, 不同数据集的信号幅度/电极阻抗差异会导致
          协方差 trace 量级差异, RPA 的 reprojection 能压缩这种差异
        - 这是 RPA 在跨数据集迁移上通常优于 RiemannianAlignment 的原因

    reference='grand_mean' 模式 (本实验未用):
        步骤 2 后额外乘以 grand_sqrt, 再做 trace 归一化.

    参考:
        Rodrigues, P. L. C., et al. "Riemannian Procrustes Analysis:
        Transfer Learning for Brain-Computer Interfaces."
        IEEE TBME, 2019.
    """
    name = 'rpa'

    def __init__(self, reference='identity', max_iter=50, tol=1e-6):
        self.reference = reference
        self.max_iter = max_iter
        self.tol = tol
        self.transforms_ = {}
        # 被试级黎曼均值缓存: {(subject, band): riem_mean}
        # 跨 fold 复用, mean_riemann 只算一次 (方案 1 优化)
        self._subj_mean_cache = {}

    def _compute_subj_mean(self, covs, subject, band):
        """计算被试级黎曼均值, 带跨 fold 缓存 (同 RiemannianAlignment)."""
        cache_key = (subject, band)
        if subject is not None and band is not None and cache_key in self._subj_mean_cache:
            return self._subj_mean_cache[cache_key]
        riem_mean = mean_riemann(covs, maxiter=self.max_iter, tol=self.tol)
        riem_mean = (riem_mean + riem_mean.T) / 2  # 对称化
        if subject is not None and band is not None:
            self._subj_mean_cache[cache_key] = riem_mean
        return riem_mean

    def fit(self, covs_per_subject, band=None):
        self.transforms_ = {}
        all_covs_list = []
        for subject, covs in covs_per_subject.items():
            riem_mean = self._compute_subj_mean(covs, subject, band)
            self.transforms_[subject] = invsqrtm(riem_mean)
            all_covs_list.append(covs)
        if self.reference == 'grand_mean':
            all_covs = np.concatenate(all_covs_list, axis=0)
            grand_riem = mean_riemann(all_covs, maxiter=self.max_iter, tol=self.tol)
            grand_riem = (grand_riem + grand_riem.T) / 2  # 对称化
            self.grand_sqrt_ = sqrtm(grand_riem)
        else:
            self.grand_sqrt_ = np.eye(covs.shape[-1])

    def transform(self, covs, subject_label=None):
        if subject_label is not None and subject_label in self.transforms_:
            T = self.transforms_[subject_label]
        else:
            G = self.grand_sqrt_
            covs_proj = G @ covs @ G
            d = covs.shape[-1]
            traces = np.trace(covs_proj, axis1=-2, axis2=-1)
            lambdas = d / traces
            return covs_proj * lambdas[:, np.newaxis, np.newaxis]
        covs_id = T @ covs @ T
        if self.reference == 'grand_mean':
            G = self.grand_sqrt_
            covs_proj = G @ covs_id @ G
        else:
            covs_proj = covs_id
        d = covs.shape[-1]
        traces = np.trace(covs_proj, axis1=-2, axis2=-1)
        lambdas = d / traces
        return covs_proj * lambdas[:, np.newaxis, np.newaxis]

    def align_subject(self, covs, subject=None, band=None):
        riem_mean = self._compute_subj_mean(covs, subject, band)
        T = invsqrtm(riem_mean)
        covs_id = T @ covs @ T
        if self.reference == 'grand_mean' and hasattr(self, 'grand_sqrt_'):
            G = self.grand_sqrt_
            covs_proj = G @ covs_id @ G
        else:
            covs_proj = covs_id
        d = covs.shape[-1]
        traces = np.trace(covs_proj, axis1=-2, axis2=-1)
        lambdas = d / traces
        return covs_proj * lambdas[:, np.newaxis, np.newaxis]


def get_aligner(name: str, **kwargs) -> BaseAlignment:
    """对齐器工厂函数."""
    registry = {
        'none': NoAlignment,
        'euclidean': EuclideanAlignment,
        'riemann': RiemannianAlignment,
        'rpa': RiemannianProcrustesAlignment,
    }
    if name not in registry:
        raise ValueError(f"未知对齐器: {name}, 可选: {list(registry.keys())}")
    return registry[name](**kwargs)


# ════════════════════════════════════════════════════════════════
# 分类评测引擎 (新增, 用于 D1 情绪分类)
# ════════════════════════════════════════════════════════════════

def evaluate_loso_classification(subjects: List[str],
                                  all_covs: Dict[str, Dict],
                                  all_y: Dict[str, np.ndarray],
                                  aligner: BaseAlignment,
                                  bands: str = "5band",
                                  classifier: str = "svm",
                                  n_features: int = 150,
                                  max_train_epochs_per_subj: int = 100,
                                  channels: str = None,
                                  verbose: bool = True) -> Dict:
    """LOSO 分类评测 (单数据集).

    Args:
        subjects: 被试 ID 列表
        all_covs: {sub: {(lo,hi): (n_epochs, C, C)}}
        all_y: {sub: (n_epochs,)} 标签
        aligner: 对齐器实例
        bands: '5band' | '8band'
        classifier: 'svm' | 'ridge'
        n_features: 特征选择数
        max_train_epochs_per_subj: 每被试训练 epoch 上限 (均匀采样, 加速 TS).
            None=用全部. SEED 831ep/被试→100 加速~8x, 结果几乎不变.
        channels: 通道配置标识 (如 '32ch'/'16ch'), 写入结果用于 T5/T6/T8 复用校验.
            None=不记录 (向后兼容).
        verbose: 是否打印进度

    Returns:
        dict: {acc_mean, acc_std, f1_mean, f1_std, acc_per_subject, ...}
    """
    freq_bands = BANDS_5 if bands == "5band" else BANDS_8

    print(f"\n{'='*60}")
    print(f"LOSO Classification + {aligner.name.upper()} Alignment")
    print(f"  Bands: {bands}, Classifier: {classifier}, n_features: {n_features}")
    print(f"  max_train_epochs_per_subj: {max_train_epochs_per_subj}")
    print(f"{'='*60}")

    acc_list, f1_list = [], []
    all_y_true, all_y_pred = [], []  # 累积所有 fold 的预测, 用于聚合混淆矩阵

    t_total_start = time.time()

    for i, test_subj in enumerate(subjects):
        t_fold = time.time()
        train_subjs = [s for s in subjects if s != test_subj]

        # 每频段: 对齐 → 池化 → 切空间
        features_list = []
        test_features_list = []

        for band in freq_bands:
            band_covs_train = {s: all_covs[s][band] for s in train_subjs}
            # 传 band 参数, 启用 aligner 的被试均值跨 fold 缓存 (方案 1 优化)
            band_covs_aligned = aligner.fit_transform(band_covs_train, band=band)

            # 子采样训练 epoch, 加速 TangentSpace (均匀采样保留标签分布)
            if max_train_epochs_per_subj is not None:
                aligned_sel = {}
                for s in train_subjs:
                    n_total = len(band_covs_aligned[s])
                    if n_total > max_train_epochs_per_subj:
                        idx = np.linspace(0, n_total - 1,
                                          max_train_epochs_per_subj, dtype=int)
                        aligned_sel[s] = band_covs_aligned[s][idx]
                    else:
                        aligned_sel[s] = band_covs_aligned[s]
            else:
                aligned_sel = band_covs_aligned

            train_covs_pooled = np.concatenate(
                [aligned_sel[s] for s in train_subjs], axis=0
            )
            # 标签也要对应子采样
            if max_train_epochs_per_subj is not None:
                labels_list = []
                for s in train_subjs:
                    n_total = len(all_y[s])
                    if n_total > max_train_epochs_per_subj:
                        idx = np.linspace(0, n_total - 1,
                                          max_train_epochs_per_subj, dtype=int)
                        labels_list.append(all_y[s][idx])
                    else:
                        labels_list.append(all_y[s])
                train_labels_pooled = np.concatenate(labels_list)
            else:
                train_labels_pooled = np.concatenate([all_y[s] for s in train_subjs])

            ts = TangentSpace(metric='riemann')
            ts_feats = ts.fit_transform(train_covs_pooled, train_labels_pooled)
            features_list.append(ts_feats)

            # 测试被试对齐 (无监督, 用全部 epoch)
            # 传 subject + band, 启用缓存 (test_subj 的 mean_riemann 在其他 fold 训练时已算过)
            test_covs = all_covs[test_subj][band]
            test_covs_aligned = aligner.align_subject(test_covs, subject=test_subj, band=band)
            test_feats = ts.transform(test_covs_aligned)
            test_features_list.append(test_feats)

        X_train = np.hstack(features_list)
        X_test = np.hstack(test_features_list)
        # y_train 用子采样后的标签 (与 X_train 行数一致)
        y_train = train_labels_pooled
        y_test = all_y[test_subj]

        # 单类保护: 训练集只有一类时, SelectKBest(f_classif) 会返回 NaN,
        # SVC/Ridge 也会退化为恒预测该类. 跳过此 fold (与 _run_itsa_baseline 一致).
        if len(np.unique(y_train)) < 2:
            if verbose:
                print(f"    [{i+1}/{len(subjects)}] {test_subj}: SKIP (训练集仅含单类)", flush=True)
            continue

        # 特征选择
        if X_train.shape[1] > n_features:
            sel = SelectKBest(f_classif, k=n_features)
            X_train = sel.fit_transform(X_train, y_train)
            X_test = sel.transform(X_test)

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 分类器
        if classifier == "svm":
            clf = SVC(kernel='rbf', C=1.0, gamma='scale')
        elif classifier == "ridge":
            clf = Ridge(alpha=1.0)
        else:
            raise ValueError(f"未知 classifier: {classifier}")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred = np.round(y_pred).astype(int) if classifier == "ridge" else y_pred

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        acc_list.append(acc)
        f1_list.append(f1)
        # 累积预测用于混淆矩阵
        all_y_true.extend(list(y_test))
        all_y_pred.extend(list(y_pred))

        if verbose:
            print(f"  [{i+1}/{len(subjects)}] {test_subj}: "
                  f"ACC={acc:.4f}, F1={f1:.4f} ({time.time()-t_fold:.1f}s)")

    t_total = time.time() - t_total_start

    # acc_list 为空的兜底: 所有 fold 都被单类保护跳过时, 返回空结果而非 NaN
    # 与 _run_itsa_baseline (run_d1_experiments.py line 715-723) 一致.
    # 实际场景下每个被试至少含 2-3 类, 不会触发, 但保留防护以应对极端调试模式.
    if len(acc_list) == 0:
        print(f"  [WARN] {aligner.name}: 所有 fold 因训练集单类被跳过, 返回空结果",
              flush=True)
        return {
            'acc_mean': 0.0, 'acc_std': 0.0,
            'acc_per_subject': [],
            'f1_mean': 0.0, 'f1_std': 0.0,
            'f1_per_subject': [],
            'n_subjects': len(subjects),
            'n_valid_folds': 0,
            'aligner': aligner.name,
            'bands': bands,
            'channels': channels,
            'classifier': classifier,
            'n_features': n_features,
            'max_train_epochs_per_subj': max_train_epochs_per_subj,
            'confusion_matrix': [],
            'confusion_matrix_labels': [],
            'time_s': float(t_total),
            'note': '所有 fold 因训练集单类被跳过',
        }

    # 聚合混淆矩阵 (所有 fold 合并)
    labels_sorted = sorted(set(all_y_true) | set(all_y_pred))
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels_sorted)

    results = {
        'acc_mean': float(np.mean(acc_list)),
        'acc_std': float(np.std(acc_list)),
        'acc_per_subject': [float(x) for x in acc_list],
        'f1_mean': float(np.mean(f1_list)),
        'f1_std': float(np.std(f1_list)),
        'f1_per_subject': [float(x) for x in f1_list],
        'n_subjects': len(subjects),
        'aligner': aligner.name,
        'bands': bands,
        'channels': channels,
        'classifier': classifier,
        'n_features': n_features,
        'max_train_epochs_per_subj': max_train_epochs_per_subj,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': [int(x) for x in labels_sorted],
        'time_s': float(t_total),
    }
    
    print(f"\n  === {aligner.name.upper()}: ACC={results['acc_mean']:.4f}±"
          f"{results['acc_std']:.4f}, F1={results['f1_mean']:.4f}±"
          f"{results['f1_std']:.4f}")
    
    return results


# ════════════════════════════════════════════════════════════════
# 跨数据集迁移评测 (新增, 用于 D1 跨库迁移)
# ════════════════════════════════════════════════════════════════

def evaluate_cross_dataset(source_covs: Dict, source_y: Dict,
                            target_covs: Dict, target_y: Dict,
                            aligner: BaseAlignment,
                            bands: str = "5band",
                            classifier: str = "svm",
                            n_features: int = 150,
                            max_train_epochs_per_subj: int = 100,
                            verbose: bool = True) -> Dict:
    """跨数据集迁移评测: 源域训练 → 目标域测试.

    Args:
        source_covs: {sub: {(lo,hi): (n_epochs, C, C)}} 源域协方差
        source_y: {sub: (n_epochs,)} 源域标签
        target_covs: {sub: {(lo,hi): (n_epochs, C, C)}} 目标域协方差
        target_y: {sub: (n_epochs,)} 目标域标签
        aligner: 对齐器实例 (通过 deepcopy 为每个 band 创建独立副本, 状态互不覆盖)
        max_train_epochs_per_subj: 源域每被试训练 epoch 上限 (均匀采样, 加速 TS).
            None=用全部. 与 T1 的优化一致, 100 ep/subj 加速 ~8x.

    Returns:
        dict: {acc_mean, f1_mean, ...}

    注意:
        每个 band 需要独立的 aligner 和 TangentSpace 实例.
        源域 fit 后, 目标域必须用同一 band 的 fit 状态做 align_subject.
        本函数为每个 band 保存 aligner 副本和 ts 实例, 避免跨 band 状态错乱.
    """
    import copy

    freq_bands = BANDS_5 if bands == "5band" else BANDS_8
    source_subs = list(source_covs.keys())
    target_subs = list(target_covs.keys())

    print(f"\n{'='*60}")
    print(f"Cross-Dataset Transfer + {aligner.name.upper()} Alignment")
    print(f"  Source: {len(source_subs)} subs, Target: {len(target_subs)} subs")
    print(f"  Bands: {bands}, Classifier: {classifier}")
    print(f"  max_train_epochs_per_subj: {max_train_epochs_per_subj}")
    print(f"{'='*60}")

    # 源域: 对齐 + 池化 + 切空间 (为每个 band 保存 aligner 副本和 ts 实例)
    features_list = []
    aligners_per_band = []  # 每个 band 的 aligner 副本 (含 fit 状态)
    ts_per_band = []        # 每个 band 的 TangentSpace 实例 (含 fit 状态)

    for band in freq_bands:
        band_covs_source = {s: source_covs[s][band] for s in source_subs}
        # 子采样源域 epoch, 加速 aligner fit + TangentSpace (与 T1 一致)
        if max_train_epochs_per_subj is not None:
            band_covs_sel = {}
            y_sel = {}
            for s in source_subs:
                n_total = len(band_covs_source[s])
                if n_total > max_train_epochs_per_subj:
                    idx = np.linspace(0, n_total - 1,
                                      max_train_epochs_per_subj, dtype=int)
                    band_covs_sel[s] = band_covs_source[s][idx]
                    y_sel[s] = source_y[s][idx]
                else:
                    band_covs_sel[s] = band_covs_source[s]
                    y_sel[s] = source_y[s]
        else:
            band_covs_sel = band_covs_source
            y_sel = source_y

        # 深拷贝 aligner, 避免各 band 互相覆盖 fit 状态
        aligner_band = copy.deepcopy(aligner)
        # 传 band 参数, 保持接口一致 (deepcopy 后 cache 为空, 此处缓存无实际加速)
        band_covs_aligned = aligner_band.fit_transform(band_covs_sel, band=band)
        train_covs_pooled = np.concatenate(
            [band_covs_aligned[s] for s in source_subs], axis=0
        )
        train_labels_pooled = np.concatenate([y_sel[s] for s in source_subs])

        ts = TangentSpace(metric='riemann')
        ts_feats = ts.fit_transform(train_covs_pooled, train_labels_pooled)
        features_list.append(ts_feats)
        aligners_per_band.append(aligner_band)
        ts_per_band.append(ts)
    
    X_train = np.hstack(features_list)
    # 注意: y_train 必须用子采样后的标签 (与 train_covs_pooled 对应)
    if max_train_epochs_per_subj is not None:
        y_train_labels = []
        for s in source_subs:
            n_total = len(source_y[s])
            if n_total > max_train_epochs_per_subj:
                idx = np.linspace(0, n_total - 1,
                                  max_train_epochs_per_subj, dtype=int)
                y_train_labels.append(source_y[s][idx])
            else:
                y_train_labels.append(source_y[s])
        y_train = np.concatenate(y_train_labels)
    else:
        y_train = np.concatenate([source_y[s] for s in source_subs])
    
    # 单类保护: 训练集只有一类时, SelectKBest(f_classif) 返回 NaN, SVC 退化.
    # 与 evaluate_loso_classification (line 451-456) 和 _run_itsa_baseline 一致.
    # 实际场景下源域多被试池化后含多类, 但保留保护以应对极端调试模式 (n_subjects=1).
    if len(np.unique(y_train)) < 2:
        print(f"  [SKIP] 源域训练集仅含单类, 跳过跨数据集评测", flush=True)
        return {
            'acc_mean': 0.0, 'acc_std': 0.0,
            'acc_per_subject': [],
            'f1_mean': 0.0, 'f1_std': 0.0,
            'f1_per_subject': [],
            'n_source': len(source_subs),
            'n_target': len(target_subs),
            'n_valid_folds': 0,
            'aligner': aligner.name,
            'bands': bands,
            'classifier': classifier,
            'transfer': 'source→target',
            'confusion_matrix': [],
            'confusion_matrix_labels': [],
            'note': '源域训练集仅含单类, 评测被跳过',
        }

    # 特征选择 + 标准化 + 分类器
    sel = None
    if X_train.shape[1] > n_features:
        sel = SelectKBest(f_classif, k=n_features)
        X_train = sel.fit_transform(X_train, y_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    if classifier == "svm":
        clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    else:
        clf = Ridge(alpha=1.0)
    clf.fit(X_train, y_train)
    
    # 目标域: 每个被试独立对齐 + 测试 (用对应 band 的 aligner 和 ts)
    acc_list, f1_list = [], []
    all_y_true, all_y_pred = [], []  # 累积预测用于混淆矩阵
    for i, t_sub in enumerate(target_subs):
        test_features_list = []
        for band_idx, band in enumerate(freq_bands):
            test_covs = target_covs[t_sub][band]
            # 用该 band 的 aligner 副本做对齐 (保持与源域一致的 fit 状态)
            # S3 修复: target subject 加 'tgt_' 前缀, 避免与 source subject 名冲突
            # 导致 _subj_mean_cache 误命中 (虽然当前 SEED/DEAP 命名不冲突, 加前缀是防护)
            t_sub_key = f"tgt_{t_sub}"
            test_covs_aligned = aligners_per_band[band_idx].align_subject(
                test_covs, subject=t_sub_key, band=band)
            # 用该 band 的 ts 做 transform
            test_feats = ts_per_band[band_idx].transform(test_covs_aligned)
            test_features_list.append(test_feats)

        X_test = np.hstack(test_features_list)
        y_test = target_y[t_sub]

        if sel is not None:
            X_test = sel.transform(X_test)
        X_test = scaler.transform(X_test)

        y_pred = clf.predict(X_test)
        y_pred = np.round(y_pred).astype(int) if classifier == "ridge" else y_pred

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        acc_list.append(acc)
        f1_list.append(f1)
        all_y_true.extend(list(y_test))
        all_y_pred.extend(list(y_pred))

        if verbose:
            print(f"  [{i+1}/{len(target_subs)}] {t_sub}: "
                  f"ACC={acc:.4f}, F1={f1:.4f}")

    # acc_list 为空的兜底: 目标域无有效被试时返回空结果而非 NaN
    # 与 evaluate_loso_classification 和 _run_itsa_baseline 一致.
    if len(acc_list) == 0:
        print(f"  [WARN] {aligner.name} Cross-Dataset: 目标域无有效评测结果, 返回空",
              flush=True)
        return {
            'acc_mean': 0.0, 'acc_std': 0.0,
            'acc_per_subject': [],
            'f1_mean': 0.0, 'f1_std': 0.0,
            'f1_per_subject': [],
            'n_source': len(source_subs),
            'n_target': len(target_subs),
            'n_valid_folds': 0,
            'aligner': aligner.name,
            'bands': bands,
            'classifier': classifier,
            'transfer': 'source→target',
            'confusion_matrix': [],
            'confusion_matrix_labels': [],
            'note': '目标域无有效评测结果',
        }

    # 聚合混淆矩阵
    labels_sorted = sorted(set(all_y_true) | set(all_y_pred))
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels_sorted)

    results = {
        'acc_mean': float(np.mean(acc_list)),
        'acc_std': float(np.std(acc_list)),
        'acc_per_subject': [float(x) for x in acc_list],
        'f1_mean': float(np.mean(f1_list)),
        'f1_std': float(np.std(f1_list)),
        'f1_per_subject': [float(x) for x in f1_list],
        'n_source': len(source_subs),
        'n_target': len(target_subs),
        'aligner': aligner.name,
        'bands': bands,
        'classifier': classifier,
        'transfer': 'source→target',
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': [int(x) for x in labels_sorted],
    }
    
    print(f"\n  === {aligner.name.upper()} Cross-Dataset: "
          f"ACC={results['acc_mean']:.4f}±{results['acc_std']:.4f}, "
          f"F1={results['f1_mean']:.4f}±{results['f1_std']:.4f}")

    return results


# ════════════════════════════════════════════════════════════════
# 统计显著性检验 (新增, P0-3)
# ════════════════════════════════════════════════════════════════

def paired_t_test(acc_a: List[float], acc_b: List[float],
                  alternative: str = "two-sided") -> Dict:
    """配对 t 检验: 比较两个对齐器在同一组被试上的 per-subject 准确率.

    Args:
        acc_a: 对齐器 A 的 per-subject 准确率列表
        acc_b: 对齐器 B 的 per-subject 准确率列表 (顺序需与 A 一致)
        alternative: 'two-sided' | 'less' | 'greater'

    Returns:
        dict: {t_stat, p_value, mean_diff, std_diff, n, alternative}
    """
    from scipy.stats import ttest_rel

    if len(acc_a) != len(acc_b):
        raise ValueError(
            f"配对样本数不一致: len(acc_a)={len(acc_a)} != len(acc_b)={len(acc_b)}. "
            "请确认两个对齐器在同一被试列表上评测."
        )
    n = len(acc_a)

    # 空列表保护: ttest_rel([], []) 返回 (nan, nan), 但下游可能误用.
    # 与 permutation_test 的空列表保护一致, 早返回带 warning 的 NaN 结果.
    if n == 0:
        return {
            't_stat': float('nan'),
            'p_value': float('nan'),
            'mean_diff': float('nan'),
            'std_diff': float('nan'),
            'n': 0,
            'alternative': alternative,
            'test': 'paired t-test (scipy ttest_rel)',
            'warning': '输入为空, 结果不可信',
        }
    a = np.asarray(acc_a, dtype=float)
    b = np.asarray(acc_b, dtype=float)
    diff = a - b

    # NaN 检查: 若输入含 NaN, ttest_rel 返回 (nan, nan), 下游会静默错误
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        return {
            't_stat': float('nan'),
            'p_value': float('nan'),
            'mean_diff': float('nan'),
            'std_diff': float('nan'),
            'n': n,
            'alternative': alternative,
            'test': 'paired t-test (scipy ttest_rel)',
            'warning': '输入含 NaN/Inf, 结果不可信',
        }

    t_stat, p_two = ttest_rel(a, b)
    # scipy 默认双侧, 单侧需转换
    if alternative == "two-sided":
        p_value = p_two
    elif alternative == "greater":  # H1: mean(a-b) > 0
        p_value = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    elif alternative == "less":     # H1: mean(a-b) < 0
        p_value = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    else:
        raise ValueError(f"未知 alternative: {alternative}")

    return {
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'mean_diff': float(np.mean(diff)),
        'std_diff': float(np.std(diff, ddof=1)),
        'n': n,
        'alternative': alternative,
        'test': 'paired t-test (scipy ttest_rel)',
    }


def permutation_test(acc_a: List[float], acc_b: List[float],
                     n_permutations: int = 10000,
                     random_seed: int = 42) -> Dict:
    """置换检验 (非参数): 比较两个对齐器的 per-subject 准确率差异.

    原假设 H0: 两个对齐器在被试上的准确率分布相同 (即差异可任意互换).
    统计量: mean(acc_a - acc_b).
    p_value = (|T_obs| <= |T_perm| 的次数) / n_permutations, 双侧.

    适用于:
        - 样本量小 (n < 30) 时正态假设不可靠
        - per-subject 准确率分布偏斜严重时
        - 审稿人要求非参数检验时

    Args:
        acc_a, acc_b: per-subject 准确率列表 (同序)
        n_permutations: 置换次数 (10000 次足够稳定)
        random_seed: 随机种子

    Returns:
        dict: {observed_diff, p_value, n_permutations, n}
    """
    if len(acc_a) != len(acc_b):
        raise ValueError(
            f"配对样本数不一致: len(acc_a)={len(acc_a)} != len(acc_b)={len(acc_b)}"
        )

    # 空列表保护: 空输入时 obs_diff/perm_stat 都是 NaN,
    # abs(NaN) >= NaN 恒为 False, count_extreme 恒为 0,
    # p_value = (0+1)/(n_perm+1) ≈ 0.0001 误判为高度显著 (falsely significant).
    # 必须早返回, 避免退化场景下 T9 报告错误的统计显著性.
    if len(acc_a) == 0:
        return {
            'observed_diff': float('nan'),
            'p_value': float('nan'),
            'n_permutations': n_permutations,
            'n': 0,
            'test': 'permutation test (sign-flipping, two-sided)',
            'warning': '输入为空, 结果不可信',
        }

    rng = np.random.RandomState(random_seed)
    a = np.asarray(acc_a, dtype=float)
    b = np.asarray(acc_b, dtype=float)
    n = len(a)

    # NaN 检查: 若输入含 NaN, obs_diff/perm_stat 都会是 NaN,
    # abs(NaN) >= NaN 永远为 False, count_extreme 恒为 0, p≈0 误判为高度显著
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        return {
            'observed_diff': float('nan'),
            'p_value': float('nan'),
            'n_permutations': n_permutations,
            'n': n,
            'test': 'permutation test (sign-flipping)',
            'warning': '输入含 NaN/Inf, 结果不可信',
        }

    # 观察到的差异 (统计量)
    obs_diff = float(np.mean(a - b))

    # 置换: 每个被试的差异符号以 0.5 概率翻转 (等价于交换 a/b)
    # T_perm = mean(s_i * (a_i - b_i)), s_i ∈ {+1, -1}
    diffs = a - b
    count_extreme = 0
    obs_abs = abs(obs_diff)
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        perm_stat = float(np.mean(signs * diffs))
        if abs(perm_stat) >= obs_abs:
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)  # +1 平滑

    return {
        'observed_diff': obs_diff,
        'p_value': float(p_value),
        'n_permutations': n_permutations,
        'n': n,
        'test': 'permutation test (sign-flipping, two-sided)',
    }


def compare_aligners(results: Dict, metric: str = "acc_per_subject",
                     reference: str = "none",
                     n_permutations: int = 10000) -> Dict:
    """对一个数据集内多个对齐器做配对统计检验 (vs 参考对齐器).

    Args:
        results: {aligner_name: {acc_per_subject: [...], ...}, ...}
                 (例如 T1 结果中某数据集的 dict)
        metric: 'acc_per_subject' | 'f1_per_subject'
        reference: 参考对齐器名 (默认 'none', 即无对齐 baseline)
        n_permutations: 置换次数

    Returns:
        dict: {aligner_name: {t_test, perm_test, delta_mean}, ...}
              不含 reference 本身
    """
    if reference not in results:
        # 找第一个可用的参考 (优先 none, 否则取第一个)
        original_ref = reference
        if "none" in results:
            reference = "none"
        else:
            reference = list(results.keys())[0]
        # 强警告: 静默 fallback 可能误导用户 (用户期望对比 X, 实际对比了 Y)
        print(f"  [stats][WARNING] 请求的参考对齐器 '{original_ref}' 不在结果中, "
              f"fallback 到 '{reference}'. 请检查数据完整性或显式指定 --reference")
        if original_ref != "none" and reference != "none":
            print(f"  [stats][WARNING] 当前 fallback 可能不是 baseline, "
                  f"统计意义可能与预期不符")

    ref_vals = results[reference][metric]
    # 空参考列表保护: 若参考对齐器 acc_per_subject 为空 (如源域单类保护触发),
    # 所有比较都无意义, 早返回避免 permutation_test 误报 falsely significant.
    # 注意: 下面的 len(cur_vals) != len(ref_vals) 对 0 != 0 返回 False 不跳过,
    # 必须在此显式检查空列表.
    if len(ref_vals) == 0:
        print(f"  [stats][WARNING] 参考对齐器 '{reference}' 的 {metric} 为空, "
              f"跳过所有比较 (可能因源域单类保护或退化结果)")
        return {}
    out = {}
    for name, res in results.items():
        if name == reference:
            continue
        if metric not in res:
            print(f"  [stats] 跳过 {name}: 无 {metric} 字段")
            continue
        cur_vals = res[metric]
        if len(cur_vals) != len(ref_vals):
            print(f"  [stats] 跳过 {name}: 样本数 {len(cur_vals)} != 参考数 {len(ref_vals)}")
            continue
        if len(cur_vals) == 0:
            print(f"  [stats] 跳过 {name}: {metric} 为空")
            continue
        out[name] = {
            't_test': paired_t_test(cur_vals, ref_vals, alternative="greater"),
            'permutation_test': permutation_test(cur_vals, ref_vals,
                                                 n_permutations=n_permutations),
            'delta_mean': float(np.mean(np.asarray(cur_vals) - np.asarray(ref_vals))),
            'reference': reference,
            'metric': metric,
        }
    return out


def compare_all_datasets(stats_results: Dict) -> Dict:
    """跨数据集聚合统计: 合并各数据集的 per-subject 差异, 做整体配对 t 检验.

    Args:
        stats_results: {dataset: {aligner: {acc_per_subject: [...]}}, ...}
                       (例如 T1 全结果)

    Returns:
        dict: {aligner: {t_test_vs_none, n_total, datasets_included}, ...}
    """
    # 收集每个对齐器相对 'none' 的所有 per-subject 差异
    deltas_by_aligner = {}  # {aligner: [diff_per_subj, ...]}
    ref_name = "none"
    for ds, ds_res in stats_results.items():
        if ref_name not in ds_res:
            continue
        ref_vals = ds_res[ref_name].get("acc_per_subject", [])
        # 空参考列表保护: ref_vals 为空时, 所有 zip(cur, ref_vals) 都产出空,
        # deltas 不会增长, 但仍会徒劳遍历. 早跳过避免无意义计算.
        if len(ref_vals) == 0:
            continue
        for name, res in ds_res.items():
            if name == ref_name:
                continue
            cur = res.get("acc_per_subject", [])
            if len(cur) != len(ref_vals):
                continue
            if len(cur) == 0:
                continue
            deltas_by_aligner.setdefault(name, []).extend(
                [float(c - r) for c, r in zip(cur, ref_vals)]
            )

    out = {}
    from scipy.stats import ttest_1samp
    for name, deltas in deltas_by_aligner.items():
        # 空/单元素 deltas 保护:
        # - n=0: ttest_1samp 返回 (nan, nan), std(ddof=1) 返回 nan+RuntimeWarning
        # - n=1: ttest_1samp 返回 (nan, nan) (std=0 除以 0), std(ddof=1) 返回 nan
        # 两种情况都会污染 T9 报告 (显示 NaN), 必须显式跳过.
        # 实际场景下 pooled n 通常 ≥ 30 (3 数据集 × 10+ 被试), 此保护仅应对退化场景.
        if len(deltas) < 2:
            print(f"  [stats][WARNING] 跳过 {name} pooled 统计: "
                  f"deltas 样本数 {len(deltas)} < 2 (可能因源域单类保护或退化结果)")
            out[name] = {
                't_stat': float('nan'),
                'p_value': float('nan'),
                'mean_delta': float(np.mean(deltas)) if len(deltas) == 1 else float('nan'),
                'std_delta': float('nan'),
                'n_total': len(deltas),
                'test': 'one-sample t-test on per-subject deltas (vs none)',
                'warning': 'deltas 样本数 < 2, 结果不可信',
            }
            continue
        deltas_arr = np.asarray(deltas)
        # 单样本 t 检验: H0: mean(deltas) = 0
        t_stat, p_val = ttest_1samp(deltas_arr, 0)
        out[name] = {
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'mean_delta': float(np.mean(deltas_arr)),
            'std_delta': float(np.std(deltas_arr, ddof=1)),
            'n_total': len(deltas),
            'test': 'one-sample t-test on per-subject deltas (vs none)',
        }
    return out
