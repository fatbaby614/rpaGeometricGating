"""
deap_loader.py — DEAP 数据集加载器
===================================

DEAP 数据结构 (已核实):
    E:\\datasets\\emotion\\DEAP\\
        data_preprocessed_python/
            s01.dat .. s32.dat     32 被试, pickle 格式
        data_preprocessed_matlab/
            s01.mat .. s32.mat      MATLAB 格式 (本加载器不用)
        data_original/              原始 BDF (本加载器不用)

数据格式 (已核实):
    每个 .dat (pickle, latin1) 包含:
        data:   (40, 40, 8064) — 40 trials × 40 channels × 8064 samples
                前 32ch 是 EEG, 后 8ch 是 peripheral (hEOG, vEOG, zEMG, tEMG, GSR, Respiration, Plethysmograph, Temperature)
        labels: (40, 4) — [valence, arousal, dominance, liking] ∈ [1, 9]

采样率: 128 Hz (已降采样, 8064 samples / 63s ≈ 128 Hz; 前 3s baseline, 后 60s 刺激)

输出统一格式:
    cov_dict: {(lo, hi): (n_epochs, C, C)} SPD
    y: (n_epochs,) 效价二分类标签 (1=高效价 >5, 0=低效价 ≤5)
    n_ch: 32 (EEG 通道数)
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from paths import DEAP_ROOT
from channel_utils import DEAP_32CH, get_channel_indices

# 兼容两种运行方式:
#   1) 作为包模块导入 (from loaders.deap_loader import ...) → 相对导入
#   2) 作为顶层脚本运行 (python code/loaders/deap_loader.py) → 绝对导入
try:
    from .seed_loader import BANDS_5, BANDS_8, _bandpass
except ImportError:
    from seed_loader import BANDS_5, BANDS_8, _bandpass

# ── 标签 ────────────────────────────────────────────────────────────
# DEAP 原标签: valence/arousal/dominance/liking ∈ [1, 9]
# 统一标签: 效价二分类 (1=高效价 >5, 0=低效价 ≤5)
# 也支持唤醒度二分类 (1=高唤醒 >5, 0=低唤醒 ≤5)
LABEL_DIMS = ['valence', 'arousal', 'dominance', 'liking']
CLASS_NAMES_VALENCE = ['low_valence', 'high_valence']

# 采样率 (已核实, 128 Hz)
SAMPLE_RATE = 128

# EEG 通道数 (前 32ch 是 EEG)
N_EEG_CH = 32


# ════════════════════════════════════════════════════════════════════
# 被试列表
# ════════════════════════════════════════════════════════════════════

def list_subjects() -> List[str]:
    """返回被试 ID 列表 (格式 's01'..'s32')."""
    pp_dir = os.path.join(DEAP_ROOT, "data_preprocessed_python")
    ids = []
    for fn in sorted(os.listdir(pp_dir)):
        if fn.endswith(".dat"):
            ids.append(fn[:-4])  # 去掉 .dat
    return ids


# ════════════════════════════════════════════════════════════════════
# 原始 EEG 加载
# ════════════════════════════════════════════════════════════════════

def load_raw_eeg(subject: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载单个被试的 raw EEG.
    
    Args:
        subject: 's01'..'s32'
    
    Returns:
        data: (40, 32, 8064) — 40 trials × 32 EEG channels × 8064 samples
        labels: (40, 4) — [valence, arousal, dominance, liking]
    """
    fp = os.path.join(DEAP_ROOT, "data_preprocessed_python", f"{subject}.dat")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"DEAP 文件不存在: {fp}")
    
    with open(fp, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    
    data = d["data"].astype(np.float64)      # (40, 40, 8064)
    labels = d["labels"].astype(np.float64)  # (40, 4)
    
    # 只取前 32ch (EEG)
    data = data[:, :N_EEG_CH, :]  # (40, 32, 8064)
    
    return data, labels


# ════════════════════════════════════════════════════════════════════
# 标签转换
# ════════════════════════════════════════════════════════════════════

def labels_to_binary(labels: np.ndarray, dim: str = "valence",
                     threshold: float = 5.0) -> np.ndarray:
    """连续标签 → 二分类标签.
    
    Args:
        labels: (n_trials, 4) [valence, arousal, dominance, liking]
        dim: 'valence' | 'arousal' | 'dominance' | 'liking'
        threshold: 阈值, >threshold → 1, ≤threshold → 0
    
    Returns:
        binary_labels: (n_trials,) 0/1
    """
    dim_idx = LABEL_DIMS.index(dim)
    return (labels[:, dim_idx] > threshold).astype(np.int64)


# ════════════════════════════════════════════════════════════════════
# 协方差计算
# ════════════════════════════════════════════════════════════════════

def precompute_subject_covs(subject: str,
                             bands: str = "5band",
                             channels: str = "32ch",
                             epoch_len_sec: float = 4.0,
                             estimator: str = "oas",
                             label_dim: str = "valence",
                             label_threshold: float = 5.0,
                             use_baseline: bool = False,
                             discard_start_sec: float = 3.0
                             ) -> Tuple[Dict[Tuple[float, float], np.ndarray],
                                         np.ndarray, int]:
    """单被试的 SPD 协方差计算.
    
    Args:
        subject: 's01'..'s32'
        bands: '5band' | '8band'
        channels: '32ch' | '16ch' | '8ch' | '4ch'
        epoch_len_sec: epoch 长度 (秒), 默认 4s
        estimator: pyriemann 协方差估计器
        label_dim: 'valence' | 'arousal' | 'dominance' | 'liking'
        label_threshold: 二分类阈值
        use_baseline: 是否用前 3s baseline 做基线校正
        discard_start_sec: trial 开头丢弃秒数 (默认 3s baseline)
    
    Returns:
        cov_dict: {(lo, hi): (n_epochs, C, C)} SPD
        y: (n_epochs,) 二分类标签 (0/1)
        n_ch: 通道数
    """
    from pyriemann.estimation import Covariances
    from channel_utils import CHANNEL_PRESETS
    
    # 加载
    data, labels = load_raw_eeg(subject)  # (40, 32, 8064), (40, 4)
    
    # 通道选择
    # ⚠️ Bug 修复: 之前 '32ch' 走特判 `list(range(32))` 返回 DEAP 原生顺序,
    # 而 SEED/SEED-IV 走 get_channel_indices(SEED_62CH, COMMON_32CH) 返回字母序.
    # 两者行列对应不同通道, 导致 T2/T3 跨库迁移的协方差矩阵数学上无意义.
    # 修复: 统一走 CHANNEL_PRESETS 分支, 用 get_channel_indices(DEAP_32CH, COMMON_32CH)
    # 得到字母序索引, 与 SEED/SEED-IV 对齐.
    if channels in CHANNEL_PRESETS:
        target_names = CHANNEL_PRESETS[channels]
        ch_indices = get_channel_indices(DEAP_32CH, target_names)
    else:
        raise ValueError(f"未知 channels: {channels}")
    
    n_ch = len(ch_indices)
    n_trials = data.shape[0]
    sr = SAMPLE_RATE
    
    # 频段
    freq_bands = BANDS_5 if bands == "5band" else BANDS_8
    
    # 标签 → 二分类
    binary_labels = labels_to_binary(labels, dim=label_dim,
                                      threshold=label_threshold)  # (40,)
    
    # 切 epoch + 滤波 + 协方差
    cov_est = Covariances(estimator=estimator)
    cov_dict = {(lo, hi): [] for (lo, hi) in freq_bands}
    all_labels = []
    
    epoch_len_samples = int(sr * epoch_len_sec)
    discard_samples = int(sr * discard_start_sec)
    
    for trial_idx in range(n_trials):
        trial_data = data[trial_idx]  # (32, 8064)
        trial_sel = trial_data[ch_indices, :]  # (C, 8064)
        
        # Baseline 校正 (可选)
        if use_baseline and discard_samples > 0:
            baseline = trial_sel[:, :discard_samples].mean(axis=1, keepdims=True)
            trial_sel = trial_sel - baseline
        
        # 丢弃 baseline 段
        if discard_samples > 0 and trial_sel.shape[1] > discard_samples:
            trial_sel = trial_sel[:, discard_samples:]
        
        # 切 epoch
        n_epochs_trial = trial_sel.shape[1] // epoch_len_samples
        if n_epochs_trial == 0:
            continue
        
        for (lo, hi) in freq_bands:
            filtered = _bandpass(trial_sel, sr, lo, hi)
            epochs = np.stack([
                filtered[:, i * epoch_len_samples:(i + 1) * epoch_len_samples]
                for i in range(n_epochs_trial)
            ], axis=0)  # (n_epochs, C, T)
            covs = cov_est.transform(epochs)
            # NaN/Inf 检查 (与 seed_loader.py 一致): OAS 估计在常数 epoch / 断线通道 /
            # 近奇异阵时可能返回 NaN, 静默污染下游. 此处只警告不 raise.
            if not np.all(np.isfinite(covs)):
                n_bad = int(np.sum(~np.isfinite(covs)))
                print(f"  [loader][WARN] {subject} band=({lo},{hi}) trial_idx={trial_idx} "
                      f"covs 含 NaN/Inf ({n_bad}/{covs.size} 元素), "
                      f"可能因常数 epoch / 断线通道 / 近奇异阵", flush=True)
            cov_dict[(lo, hi)].append(covs)
        
        all_labels.extend([binary_labels[trial_idx]] * n_epochs_trial)
    
    y = np.array(all_labels, dtype=np.int64)
    for band in freq_bands:
        cov_dict[band] = np.concatenate(cov_dict[band], axis=0)
    
    return cov_dict, y, n_ch


def precompute_all_deap(subjects: Optional[List[str]] = None,
                         bands: str = "5band",
                         channels: str = "32ch",
                         epoch_len_sec: float = 4.0,
                         estimator: str = "oas",
                         label_dim: str = "valence",
                         label_threshold: float = 5.0,
                         verbose: bool = True,
                         use_cache: bool = True,
                         ) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    """预计算所有被试的 SPD.

    Args:
        use_cache: 是否使用磁盘缓存 (默认 True, 避免重复计算)
    """
    if subjects is None:
        subjects = list_subjects()

    from cov_cache import cached_precompute

    covs, y = {}, {}
    for i, sub in enumerate(subjects):
        try:
            cd, yi, nc = cached_precompute(
                "DEAP", sub, precompute_subject_covs,
                use_cache=use_cache,
                bands=bands, channels=channels,
                epoch_len_sec=epoch_len_sec, estimator=estimator,
                label_dim=label_dim, label_threshold=label_threshold,
                discard_start_sec=3.0,
            )
            covs[sub] = cd
            y[sub] = yi
            if verbose:
                print(f"  [{i+1}/{len(subjects)}] {sub}: {len(yi)} epochs × {nc}ch, "
                      f"label dist = {np.bincount(yi, minlength=2)}")
        except Exception as e:
            print(f"  [SKIP] {sub}: {e}")

    return covs, y


# ════════════════════════════════════════════════════════════════════
# 自测
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    subs = list_subjects()
    print(f"Total subjects: {len(subs)}")
    print(f"First 5: {subs[:5]}")
    
    print(f"\n--- Quick test on s01 (32ch, 5band, valence) ---")
    sub = subs[0]
    cd, yi, nc = precompute_subject_covs(sub, bands="5band", channels="32ch",
                                          label_dim="valence")
    print(f"sub: {sub}, n_channels={nc}, n_epochs={len(yi)}")
    print(f"label distribution (valence): {np.bincount(yi, minlength=2)}")
    for b, c in cd.items():
        print(f"  band {b} Hz: cov shape = {c.shape}")
    
    print(f"\n--- 16ch wearable test ---")
    cd16, yi16, nc16 = precompute_subject_covs(sub, bands="5band", channels="16ch")
    print(f"  n_channels={nc16}, n_epochs={len(yi16)}")
