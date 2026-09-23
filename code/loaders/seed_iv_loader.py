"""
seed_iv_loader.py — SEED-IV 数据集加载器
=========================================

SEED-IV 数据结构 (已核实):
    E:\\datasets\\emotion\\SEED_IV\\
        eeg_raw_data/
            1/  {sub}_{date}.mat     15被试, session 1
            2/  {sub}_{date}.mat     session 2
            3/  {sub}_{date}.mat     session 3
        ReadMe.txt
        Channel Order.xlsx

数据格式 (已核实):
    每个 .mat 文件包含 24 个 trial (key: {prefix}_eeg1 .. {prefix}_eeg24)
    每个 trial: (62, T) — 62 通道, T 时间点 (200Hz 采样)

标签 (已核实, ReadMe.txt):
    session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
    session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
    session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    0=neutral, 1=sad, 2=fear, 3=happy

输出统一格式:
    cov_dict: {(lo, hi): (n_epochs, C, C)} SPD
    y: (n_epochs,) 标签 (0=中/1=悲/2=惧/3=喜)
"""
from __future__ import annotations

import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from paths import SEED_IV_ROOT
from channel_utils import SEED_62CH, get_channel_indices

# 兼容两种运行方式:
#   1) 作为包模块导入 (from loaders.seed_iv_loader import ...) → 相对导入
#   2) 作为顶层脚本运行 (python code/loaders/seed_iv_loader.py) → 绝对导入
try:
    from .seed_loader import (
        BANDS_5, BANDS_8, SAMPLE_RATE,
        _bandpass, _resample, _get_trial_keys
    )
except ImportError:
    from seed_loader import (
        BANDS_5, BANDS_8, SAMPLE_RATE,
        _bandpass, _resample, _get_trial_keys
    )

# ── 标签 (已核实) ──────────────────────────────────────────────────
# 0=neutral, 1=sad, 2=fear, 3=happy (与 SEED-IV ReadMe 一致)
SESSION_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}
CLASS_NAMES = ['neutral', 'sad', 'fear', 'happy']


# ════════════════════════════════════════════════════════════════════
# 被试列表
# ════════════════════════════════════════════════════════════════════

def list_subjects() -> List[str]:
    """返回被试 session ID 列表 (格式 '{session}/{sub}_{date}')."""
    ids = []
    raw_dir = os.path.join(SEED_IV_ROOT, "eeg_raw_data")
    for session in [1, 2, 3]:
        session_dir = os.path.join(raw_dir, str(session))
        if not os.path.isdir(session_dir):
            continue
        for fn in sorted(os.listdir(session_dir)):
            if fn.endswith(".mat"):
                ids.append(f"s{session}/{fn[:-4]}")
    return ids


# ════════════════════════════════════════════════════════════════════
# 原始 EEG 加载
# ════════════════════════════════════════════════════════════════════

def _parse_session(subject: str) -> int:
    """从 's1/1_20160518' 解析 session 号."""
    m = re.match(r's(\d+)/', subject)
    return int(m.group(1)) if m else 1


def load_raw_eeg(subject: str) -> Tuple[List[np.ndarray], List[int]]:
    """加载单个 session 的 raw EEG.
    
    Args:
        subject: 's{session}/{sub}_{date}' (如 's1/1_20160518')
    
    Returns:
        trials: list of (62, T_i)
        labels: list of int (0/1/2/3)
    """
    import scipy.io as sio
    
    session = _parse_session(subject)
    sub_id = subject.split("/")[1]
    fp = os.path.join(SEED_IV_ROOT, "eeg_raw_data", str(session), f"{sub_id}.mat")
    
    if not os.path.exists(fp):
        raise FileNotFoundError(f"SEED-IV 文件不存在: {fp}")
    
    m = sio.loadmat(fp)
    trial_keys = _get_trial_keys(m)
    
    if len(trial_keys) != 24:
        warnings.warn(f"{subject}: 预期 24 trials, 实际 {len(trial_keys)}")
    
    trials = []
    labels = []
    session_label = SESSION_LABELS[session]
    
    for i, k in enumerate(trial_keys[:24]):
        data = m[k]  # (62, T)
        trials.append(data.astype(np.float64))
        labels.append(session_label[i])
    
    return trials, labels


# ════════════════════════════════════════════════════════════════════
# 协方差计算 (复用 seed_loader 的逻辑)
# ════════════════════════════════════════════════════════════════════

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
    
    与 seed_loader.precompute_subject_covs 接口一致, 仅标签不同 (4类)。
    """
    from pyriemann.estimation import Covariances
    from channel_utils import CHANNEL_PRESETS
    
    trials, trial_labels = load_raw_eeg(subject)
    
    # 通道选择
    if channels == '62ch':
        ch_indices = list(range(62))
    elif channels in CHANNEL_PRESETS:
        target_names = CHANNEL_PRESETS[channels]
        ch_indices = get_channel_indices(SEED_62CH, target_names)
    else:
        raise ValueError(f"未知 channels: {channels}")
    
    n_ch = len(ch_indices)
    freq_bands = BANDS_5 if bands == "5band" else BANDS_8
    sr = SAMPLE_RATE
    
    cov_est = Covariances(estimator=estimator)
    cov_dict = {(lo, hi): [] for (lo, hi) in freq_bands}
    all_labels = []
    
    epoch_len_samples = int(target_sr * epoch_len_sec)
    discard_samples = int(target_sr * discard_start_sec)
    
    for trial_data, label in zip(trials, trial_labels):
        trial_sel = trial_data[ch_indices, :]
        trial_resampled = _resample(trial_sel, sr, target_sr)
        
        if discard_samples > 0 and trial_resampled.shape[1] > discard_samples:
            trial_resampled = trial_resampled[:, discard_samples:]
        
        n_epochs_trial = trial_resampled.shape[1] // epoch_len_samples
        if n_epochs_trial == 0:
            continue
        
        for (lo, hi) in freq_bands:
            filtered = _bandpass(trial_resampled, target_sr, lo, hi)
            epochs = np.stack([
                filtered[:, i * epoch_len_samples:(i + 1) * epoch_len_samples]
                for i in range(n_epochs_trial)
            ], axis=0)
            covs = cov_est.transform(epochs)
            # NaN/Inf 检查 (与 seed_loader.py 一致): OAS 估计在常数 epoch / 断线通道 /
            # 近奇异阵时可能返回 NaN, 静默污染下游. 此处只警告不 raise.
            if not np.all(np.isfinite(covs)):
                n_bad = int(np.sum(~np.isfinite(covs)))
                print(f"  [loader][WARN] {subject} band=({lo},{hi}) "
                      f"covs 含 NaN/Inf ({n_bad}/{covs.size} 元素), "
                      f"可能因常数 epoch / 断线通道 / 近奇异阵", flush=True)
            cov_dict[(lo, hi)].append(covs)
        
        all_labels.extend([label] * n_epochs_trial)
    
    y = np.array(all_labels, dtype=np.int64)
    for band in freq_bands:
        cov_dict[band] = np.concatenate(cov_dict[band], axis=0)
    
    return cov_dict, y, n_ch


def precompute_all_seed_iv(subjects: Optional[List[str]] = None,
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
    """
    if subjects is None:
        subjects = list_subjects()

    from cov_cache import cached_precompute

    covs, y = {}, {}
    for i, sub in enumerate(subjects):
        try:
            cd, yi, nc = cached_precompute(
                "SEED_IV", sub, precompute_subject_covs,
                use_cache=use_cache,
                bands=bands, channels=channels,
                epoch_len_sec=epoch_len_sec, estimator=estimator,
                target_sr=target_sr,
                discard_start_sec=3.0,
            )
            covs[sub] = cd
            y[sub] = yi
            if verbose:
                print(f"  [{i+1}/{len(subjects)}] {sub}: {len(yi)} epochs × {nc}ch, "
                      f"label dist = {np.bincount(yi, minlength=4)}")
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
    print(f"Class names: {CLASS_NAMES}")
    
    print(f"\n--- Quick test on first session (32ch, 5band) ---")
    sub = subs[0]
    cd, yi, nc = precompute_subject_covs(sub, bands="5band", channels="32ch")
    print(f"sub: {sub}, n_channels={nc}, n_epochs={len(yi)}")
    print(f"label distribution: {np.bincount(yi, minlength=4)}")
    for b, c in cd.items():
        print(f"  band {b} Hz: cov shape = {c.shape}")
