"""
seed_loader.py — SEED 数据集加载器
===================================

SEED 数据结构 (已核实):
    E:\\datasets\\emotion\\SEED\\SEED\\
        Preprocessed_EEG/
            {sub}_{date}.mat     15被试 × 3 sessions = 45 文件
            label.mat            全局标签 [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
            readme.txt
        channel-order.xlsx        62ch 通道顺序
        ExtractedFeatures/        预提取 DE/PSD 特征

数据格式 (已核实):
    每个 .mat 文件包含 15 个 trial (key: {prefix}_eeg1 .. {prefix}_eeg15)
    每个 trial: (62, T) — 62 通道, T 时间点 (200Hz 采样)
    T 因 trial 长度不同而异 (约 47000-70000 点)

标签 (已核实, label.mat):
    label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    1 = positive, 0 = neutral, -1 = negative

输出统一格式:
    cov_dict: {(lo, hi): (n_epochs, C, C)} SPD 协方差矩阵
    y: (n_epochs,) 标签 (统一为 2=正/1=中/0=负)
    n_ch: 使用的通道数
"""
from __future__ import annotations

import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── 路径 (跨平台) ──────────────────────────────────────────────────
from paths import SEED_ROOT

# ── 通道定义 ────────────────────────────────────────────────────────
from channel_utils import SEED_62CH, get_channel_indices, COMMON_32CH

# ── 频段 (与 ric_da_core.py 保持一致) ────────────────────────────
# 注意: 此处 BANDS_5/BANDS_8 必须与 ric_da_core.py 中的定义完全一致,
# 否则 evaluate_loso_classification 遍历 freq_bands 时会因 key 不匹配而 KeyError.
BANDS_5 = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]
# 8 频段细分 (标准分解): δ / θ / low-α / high-α / low-β / high-β / low-γ / high-γ
BANDS_8 = [(1, 4), (4, 8), (8, 11), (11, 14),
           (14, 20), (20, 31), (31, 40), (40, 50)]

# ── 标签 (已核实) ──────────────────────────────────────────────────
# SEED 原标签: 1=positive, 0=neutral, -1=negative
# 统一标签: 2=positive, 1=neutral, 0=negative (便于与 SEED-IV 对齐)
SEED_LABEL_ORIGINAL = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
SEED_LABEL_UNIFIED = [2 if l == 1 else 1 if l == 0 else 0 for l in SEED_LABEL_ORIGINAL]
# 类名
CLASS_NAMES = ['negative', 'neutral', 'positive']  # 索引 0/1/2

# 采样率 (已核实)
SAMPLE_RATE = 200


# ════════════════════════════════════════════════════════════════════
# 被试列表
# ════════════════════════════════════════════════════════════════════

def list_subjects() -> List[str]:
    """返回被试 ID 列表 (格式 '{sub}_{date}').
    
    SEED 有 15 被试 × 3 sessions = 45 文件。
    每个文件作为独立"被试 session"处理 (与 BCMI 惯例一致)。
    """
    preprocessed_dir = os.path.join(SEED_ROOT, "Preprocessed_EEG")
    ids = []
    for fn in sorted(os.listdir(preprocessed_dir)):
        if fn.endswith(".mat") and not fn.startswith("label"):
            ids.append(fn[:-4])  # 去掉 .mat
    return ids


def list_sessions_per_subject() -> Dict[str, List[str]]:
    """按被试 ID 分组: {'1': ['1_20131027', '1_20131030', '1_20131107'], ...}"""
    subjects = list_subjects()
    grouped = {}
    for sid in subjects:
        sub_num = sid.split("_")[0]
        grouped.setdefault(sub_num, []).append(sid)
    return grouped


# ════════════════════════════════════════════════════════════════════
# 原始 EEG 加载
# ════════════════════════════════════════════════════════════════════

def _get_trial_keys(mat_data: dict) -> List[str]:
    """从 .mat 字典中提取 trial key (按 eeg1..eeg15 顺序)."""
    keys = [k for k in mat_data.keys() if not k.startswith("__") and "eeg" in k.lower()]
    # 按 eeg1, eeg2, ..., eeg15 排序
    def sort_key(k):
        m = re.search(r'eeg(\d+)', k.lower())
        return int(m.group(1)) if m else 999
    return sorted(keys, key=sort_key)


def load_raw_eeg(subject: str) -> Tuple[List[np.ndarray], List[int]]:
    """加载单个 session 的 raw EEG.
    
    Args:
        subject: 被试 session ID (如 '1_20131027')
    
    Returns:
        trials: list of (62, T_i) — 每个 trial 的 raw EEG
        labels: list of int — 每个 trial 的统一标签 (0/1/2)
    """
    import scipy.io as sio
    
    fp = os.path.join(SEED_ROOT, "Preprocessed_EEG", f"{subject}.mat")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"SEED 文件不存在: {fp}")
    
    m = sio.loadmat(fp)
    trial_keys = _get_trial_keys(m)
    
    if len(trial_keys) != 15:
        warnings.warn(f"{subject}: 预期 15 trials, 实际 {len(trial_keys)}")
    
    trials = []
    labels = []
    for i, k in enumerate(trial_keys[:15]):  # 只取前 15 个
        data = m[k]  # (62, T)
        trials.append(data.astype(np.float64))
        labels.append(SEED_LABEL_UNIFIED[i])
    
    return trials, labels


# ════════════════════════════════════════════════════════════════════
# 协方差计算
# ════════════════════════════════════════════════════════════════════

def _bandpass(data: np.ndarray, sr: float, lo: float, hi: float) -> np.ndarray:
    """零相位 Butterworth 4 阶带通滤波. data: (C, T)."""
    from scipy.signal import butter, filtfilt
    nyq = sr / 2.0
    lo_n = max(lo, 0.5) / nyq
    hi_n = min(hi, nyq - 1.0) / nyq
    if hi_n <= lo_n:
        return data
    b, a = butter(4, [lo_n, hi_n], btype="band")
    padlen = min(3 * max(len(a), len(b)), data.shape[-1] - 1)
    return filtfilt(b, a, data, axis=-1, padlen=padlen).astype(np.float64)


def _resample(data: np.ndarray, sr_orig: float, sr_target: float) -> np.ndarray:
    """降采样. data: (C, T)."""
    if sr_orig == sr_target:
        return data
    from scipy.signal import resample_poly
    # 找最简整数比
    from math import gcd
    g = gcd(int(sr_orig), int(sr_target))
    up = int(sr_target) // g
    down = int(sr_orig) // g
    return resample_poly(data, up, down, axis=-1).astype(np.float64)


def precompute_subject_covs(subject: str,
                             bands: str = "5band",
                             channels: str = "32ch",
                             epoch_len_sec: float = 4.0,
                             estimator: str = "oas",
                             target_sr: float = 128.0,
                             discard_start_sec: float = 3.0
                             ) -> Tuple[Dict[Tuple[float, float], np.ndarray],
                                         np.ndarray, int]:
    """单被试 session 的 SPD 协方差计算.
    
    Args:
        subject: 被 session ID
        bands: '5band' | '8band'
        channels: '32ch' (公共子集) | '62ch' (全通道) | '16ch' | '8ch' | '4ch'
        epoch_len_sec: epoch 长度 (秒), 默认 4s
        estimator: pyriemann 协方差估计器
        target_sr: 目标采样率 (统一降采样到 128Hz)
        discard_start_sec: trial 开头丢弃秒数 (缓冲)
    
    Returns:
        cov_dict: {(lo, hi): (n_epochs, C, C)} SPD
        y: (n_epochs,) 统一标签 (0=负/1=中/2=正)
        n_ch: 通道数
    """
    from pyriemann.estimation import Covariances
    from channel_utils import CHANNEL_PRESETS
    
    # 加载 raw EEG
    trials, trial_labels = load_raw_eeg(subject)
    
    # 通道选择
    if channels == '62ch':
        ch_indices = list(range(62))
        ch_sel = SEED_62CH
    elif channels in CHANNEL_PRESETS:
        target_names = CHANNEL_PRESETS[channels]
        ch_indices = get_channel_indices(SEED_62CH, target_names)
        ch_sel = [SEED_62CH[i] for i in ch_indices]
    else:
        raise ValueError(f"未知 channels: {channels}")
    
    n_ch = len(ch_indices)
    
    # 频段
    freq_bands = BANDS_5 if bands == "5band" else BANDS_8
    
    # 降采样
    sr = SAMPLE_RATE
    
    # 切 epoch + 滤波 + 协方差
    cov_est = Covariances(estimator=estimator)
    cov_dict = {(lo, hi): [] for (lo, hi) in freq_bands}
    all_labels = []
    
    epoch_len_samples = int(target_sr * epoch_len_sec)
    discard_samples = int(target_sr * discard_start_sec)
    
    for trial_data, label in zip(trials, trial_labels):
        # 选通道 + 降采样
        trial_sel = trial_data[ch_indices, :]  # (C, T)
        trial_resampled = _resample(trial_sel, sr, target_sr)  # (C, T')
        
        # 丢弃开头缓冲
        if discard_samples > 0 and trial_resampled.shape[1] > discard_samples:
            trial_resampled = trial_resampled[:, discard_samples:]
        
        # 切 epoch
        n_epochs_trial = trial_resampled.shape[1] // epoch_len_samples
        if n_epochs_trial == 0:
            continue
        
        for (lo, hi) in freq_bands:
            # 滤波 (在降采样后的信号上)
            filtered = _bandpass(trial_resampled, target_sr, lo, hi)
            
            # 切 epoch 并转置为 (n_epochs, C, T)
            epochs = np.stack([
                filtered[:, i * epoch_len_samples:(i + 1) * epoch_len_samples]
                for i in range(n_epochs_trial)
            ], axis=0)  # (n_epochs, C, T)
            
            covs = cov_est.transform(epochs)  # (n_epochs, C, C)
            # NaN/Inf 检查: OAS 估计在常数 epoch / 断线通道 / 近奇异阵时可能返回 NaN.
            # NaN 进入缓存后会静默污染下游所有评测 (TS+ SVM, Fréchet, DL baseline).
            # 此处只警告不 raise, 让下游 (DL baseline _normalize_covs, frechet_ci)
            # 的 NaN 检查决定是否跳过该被试. raise 会丢失整个被试, 警告保留数据.
            if not np.all(np.isfinite(covs)):
                n_bad = int(np.sum(~np.isfinite(covs)))
                print(f"  [loader][WARN] {subject} band=({lo},{hi}) "
                      f"covs 含 NaN/Inf ({n_bad}/{covs.size} 元素), "
                      f"可能因常数 epoch / 断线通道 / 近奇异阵", flush=True)
            cov_dict[(lo, hi)].append(covs)
        
        all_labels.extend([label] * n_epochs_trial)
    
    # 拼接所有 trial
    y = np.array(all_labels, dtype=np.int64)
    for band in freq_bands:
        cov_dict[band] = np.concatenate(cov_dict[band], axis=0)
    
    return cov_dict, y, n_ch


def precompute_all_seed(subjects: Optional[List[str]] = None,
                         bands: str = "5band",
                         channels: str = "32ch",
                         epoch_len_sec: float = 4.0,
                         estimator: str = "oas",
                         target_sr: float = 128.0,
                         verbose: bool = True,
                         use_cache: bool = True,
                         ) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    """预计算所有被试的 SPD.

    Args:
        use_cache: 是否使用磁盘缓存 (默认 True, 避免重复计算)
    Returns:
        covs: {sub_session: {(lo,hi): (n_epochs, C, C)}}
        y:    {sub_session: (n_epochs,)} 统一标签
    """
    if subjects is None:
        subjects = list_subjects()

    # 接入磁盘缓存
    from cov_cache import cached_precompute

    covs, y = {}, {}
    for i, sub in enumerate(subjects):
        try:
            cd, yi, nc = cached_precompute(
                "SEED", sub, precompute_subject_covs,
                use_cache=use_cache,
                bands=bands, channels=channels,
                epoch_len_sec=epoch_len_sec, estimator=estimator,
                target_sr=target_sr,
                discard_start_sec=3.0,  # 默认值, 加入缓存 key
            )
            covs[sub] = cd
            y[sub] = yi
            if verbose:
                print(f"  [{i+1}/{len(subjects)}] {sub}: {len(yi)} epochs × {nc}ch, "
                      f"label dist = {np.bincount(yi, minlength=3)}")
        except Exception as e:
            print(f"  [SKIP] {sub}: {e}")

    return covs, y


# ════════════════════════════════════════════════════════════════════
# 自测
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    subs = list_subjects()
    print(f"Total subject-sessions: {len(subs)}")
    print(f"First 5: {subs[:5]}")
    print(f"Labels (unified): {SEED_LABEL_UNIFIED}")
    print(f"Class names: {CLASS_NAMES}")
    
    print(f"\n--- Quick test on first session (32ch, 5band) ---")
    sub = subs[0]
    cd, yi, nc = precompute_subject_covs(sub, bands="5band", channels="32ch")
    print(f"sub: {sub}, n_channels={nc}, n_epochs={len(yi)}")
    print(f"label distribution: {np.bincount(yi, minlength=3)}")
    for b, c in cd.items():
        print(f"  band {b} Hz: cov shape = {c.shape}, "
              f"trace mean = {np.trace(c, axis1=-2, axis2=-1).mean():.2f}")
