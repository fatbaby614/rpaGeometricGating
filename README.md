# Label-Free Geometric Gating for Subject-Adaptive Riemannian Alignment in EEG Analysis

Reference implementation and experiment pipeline for the manuscript submitted to
*Journal of Neuroscience Methods* (Elsevier).

This repository provides a **geometry-aware framework** for assessing and automating
Riemannian alignment in EEG-based emotion recognition. It quantifies *when* alignment
— in particular Riemannian Procrustes Alignment (RPA) — is reliable, and it supplies a
**label-free** rule that selects an alignment arm for each incoming recording
**before deployment**, using no target labels.

---

## Highlights

- **Centering Index (CI).** A Fréchet-distance-based, label-free diagnostic of
  within-dataset centering. `CI < 0.5` flags effective alignment; RPA reaches
  `CI < 0.5` across all 15 band–dataset combinations studied here.
- **Transferability ratio.** The cross-dataset to cross-subject Fréchet ratio
  `rho = d_ds / d_subj` pre-screens cross-dataset feasibility: `rho < 1` suggests
  applicability, `rho > 2` indicates a high risk of breakdown.
- **Label-free gating.** Extended to single-recording resolution, a rule on the
  band-averaged ratio matches the best fixed alignment arm within `0.5` points in
  every held-out rotation and significantly outperforms the inferior arms — without
  any target labels.
- **Wearable robustness.** An 8-channel montage preserves `96.4%` of the 32-channel
  RPA accuracy, and 16 channels preserve `98.3%`.

---

## Method at a glance

EEG trials are represented as symmetric positive-definite (SPD) covariance matrices.
Four alignment arms are supported and compared under identical evaluation:

| Arm | Key | Description |
| --- | --- | --- |
| No alignment | `none` | Baseline, no alignment |
| Euclidean Alignment | `euclidean` | Re-centers covariances in the ambient space (He et al.) |
| Riemannian Alignment | `riemann` | Parallel transport on the SPD manifold (Zanini et al.) |
| Riemannian Procrustes Alignment | `rpa` | Procrustes-style orthogonal transport (Rodrigues et al.) |

Geometric quantification uses the affine-invariant Riemannian metric (AIRM) Fréchet
distance and the Centering Index `CI = d_after / d_before`.

---

## Repository layout

```
code/
  run_all.py                 # one-command launcher for the full D1 experiment matrix
  ric_da_core.py             # aligner classes + LOSO / cross-dataset evaluation engine
  frechet_ci.py              # Fréchet distance matrices + Centering Index
  channel_utils.py           # per-dataset channel definitions and common-montage mapping
  cov_cache.py               # on-disk covariance cache (avoids recomputation)
  paths.py                   # cross-platform dataset path resolution
  _smoke_test.py             # dependency / module smoke test
  experiments/
    run_d1_experiments.py    # T1–T9 experiment driver
    ablation_experiments.py  # delta-band removal, metric, reference, CI-threshold ablations
  gating/
    subject_geometry.py      # per-fold label-free geometry (rho_hat, Delta_CI)
    threshold_calibration.py # threshold calibration + leave-one-dataset-out validation
    gating_analysis.py       # gating analysis and figures
  baselines/
    base.py                  # DL baseline base class + factory
    dgcnn_spd.py             # DGCNN (SPD-domain variant)
    mdjpt_wrapper.py         # mdJPT wrapper (original repo or lightweight fallback)
  loaders/
    seed_loader.py           # SEED loader
    seed_iv_loader.py        # SEED-IV loader
    deap_loader.py           # DEAP loader
cache/                       # auto-generated covariance cache ({dataset}_{subject}_{hash}.npz)
results/                     # auto-generated experiment outputs (d1_*.json, G1_*.json, ...)
```

`cache/` and `results/` are created automatically at runtime and are not tracked.

---

## Requirements

- Python 3.9+ (tested on 3.10 and 3.13)
- Core dependencies:
  - `numpy`
  - `scipy`
  - `scikit-learn`
  - `pyriemann`
- Optional:
  - `matplotlib` — for the gating figures in `code/gating/gating_analysis.py`
  - `torch` — only for the locally trained deep baselines in `code/baselines/`

Install the core stack:

```bash
pip install numpy scipy scikit-learn pyriemann
```

---

## Datasets

This repository does **not** redistribute any data. Three public datasets are used:

| Dataset | Classes used | Samples | Channels (used) | Source |
| --- | --- | --- | --- | --- |
| SEED | 3-class (positive / neutral / negative) | 15 subjects × 3 sessions | 62 → 32 common | BCMI, SJTU — http://bcmi.sjtu.edu.cn/~seed/ |
| SEED-IV | 4-class | 15 subjects × 3 sessions | 62 → 32 common | BCMI, SJTU — http://bcmi.sjtu.edu.cn/~seed/ |
| DEAP | 2-class (valence) | 32 subjects | 32 (EEG) | QMUL — https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ |

Please obtain each dataset from its official distributor and comply with the
respective license terms.

---

## Configuring data paths

Paths are resolved by `code/paths.py` in the following priority order:

1. The in-code constant `DATA_ROOT` at the top of `code/paths.py` (recommended).
2. Per-dataset environment variables (`SEED_ROOT`, `SEED_IV_ROOT`, `DEAP_ROOT`, `DROZY_ROOT`, `SEED_VIG_ROOT`).
3. The generic root environment variable `RIEMANN_EMO_DATA_ROOT`.
4. OS defaults (Windows: `E:\datasets\emotion\...`; Linux: `/mnt/data1/home/tanhuang/datasets/...`).

Either set `DATA_ROOT` in `code/paths.py`:

```python
DATA_ROOT = r"E:\datasets\emotion"      # Windows
# DATA_ROOT = "/mnt/data1/home/tanhuang/datasets"   # Linux
```

or export a root directory:

```bash
# Windows (PowerShell)
$env:RIEMANN_EMO_DATA_ROOT = "E:\datasets\emotion"
# Linux
export RIEMANN_EMO_DATA_ROOT=/mnt/data1/home/tanhuang/datasets
```

Expected on-disk layout:

```
<DATA_ROOT>/
  SEED/SEED/Preprocessed_EEG/{sub}_{date}.mat, label.mat, ...
  SEED_IV/{sub}_{session}.mat, label.mat, ...
  DEAP/data_preprocessed_python/s01.dat ... s32.dat
```

Verify resolution with:

```bash
python code/paths.py
```

---

## Quick start

```bash
# 1) Verify the environment and module imports (no data needed)
python code/_smoke_test.py

# 2) Optional: also test real data reading for the first subject of each dataset
python code/_smoke_test.py --data

# 3) Fast run: first 3 subjects, SEED only, table T1
python code/run_all.py --quick --tables T1 --datasets SEED

# 4) Full experiment matrix (all tables, all datasets)
python code/run_all.py
```

The launcher `code/run_all.py` checks the environment, then delegates to
`code/experiments/run_d1_experiments.py`. You can also call the driver directly:

```bash
python code/experiments/run_d1_experiments.py --tables T1,T7 --datasets SEED --channels 32ch --bands 5band
```

---

## Experiment tables

| Table | Description |
| --- | --- |
| `T1` | Single-dataset leave-one-session-out baseline (SEED / SEED-IV / DEAP) |
| `T2` | Cross-dataset transfer SEED → DEAP |
| `T3` | Cross-dataset transfer DEAP → SEED |
| `T4` | Same-device cross-label transfer SEED ↔ SEED-IV (binary) |
| `T4mc` | SEED ↔ SEED-IV multi-class variant |
| `T5` | Channel sensitivity (32 / 16 / 8 / 4 channels) |
| `T6` | Band sensitivity (5-band vs 8-band) |
| `T7` | Fréchet distance + Centering Index geometric quantification |
| `T8` | Baseline comparison (EA, ITSA, Riemannian; optional DGCNN-SPD / mdJPT) |
| `T9` | Statistical significance tests (paired t-test, permutation test) |

Aligners are selected with `--aligners none,euclidean,riemann,rpa`; channel
configuration with `--channels 32ch|16ch|8ch|4ch`; band configuration with
`--bands 5band|8band`.

Optional deep baselines for `T8`:

```bash
# Locally trained (requires torch); mdJPT also needs MDJPT_ROOT for the original repo:
export MDJPT_ROOT=/path/to/mdjpt
python code/experiments/run_d1_experiments.py --tables T8 --datasets SEED --dl_baselines DGCNN-SPD,mdJPT
```

---

## Label-free gating pipeline

The gating analysis is a three-step pipeline under `code/gating/`:

1. **`subject_geometry.py`** — computes per-fold, label-free geometric quantities
   (`rho_hat`, `Delta_CI`) and joins them with the `T1` accuracy results.
   ```bash
   python code/gating/subject_geometry.py --datasets SEED,SEED_IV,DEAP --channels 32ch
   ```
2. **`threshold_calibration.py`** — calibrates the gating thresholds with
   leave-one-dataset-out (LODO) validation and reports Spearman correlation,
   ROC-AUC, and interval-policy grids.
   ```bash
   python code/gating/threshold_calibration.py
   ```
3. **`gating_analysis.py`** — produces the gating analysis and summary figures.

All gating decisions are made from unlabeled data only; accuracy labels are used
solely for offline calibration and validation.

---

## Outputs

- `results/d1_*.json` — raw numeric results for each experiment table.
- `results/G1_subject_geometry_*.json` — per-fold label-free geometry.
- `results/ablation_*_*.json` — ablation outputs.
- `cache/*.npz` — cached covariance matrices, keyed by a hash of all
  output-affecting parameters (`CACHE_VERSION` in `code/cov_cache.py`).

Results and cache files are written with timestamps and the driver saves
incrementally, so long runs survive interruption.

---

## Reproducibility notes

- **Evaluation protocol.** Cross-dataset and within-dataset folds are *recording-unit*
  (participant–session) based; unless stated otherwise, "cross-subject" in the driver
  means generalization across recording units and the terms `cross-subject`,
  `cross-recording`, and `session-level` are interchangeable. This is session-level
  generalization, not strictly participant-disjoint (LOSO) generalization.
- **Splits.** Data splits are deterministic and derived from the sorted subject /
  session lists in the loaders, so re-running reproduces the same folds.
- **Random seeds.** Deep-learning baselines fix the global seed at the start of
  `evaluate_loso()` (`code/baselines/base.py`), and statistical routines use a fixed
  RNG seed (`code/gating/threshold_calibration.py`).
- **Caching.** Changing band definitions, channel configurations, or label mappings
  invalidates the cache via `CACHE_VERSION`; older caches are recomputed.

---

## Citation

If you use this code, please cite the accompanying manuscript:

```bibtex
@article{tan2026geometricgating,
  title   = {Label-Free Geometric Gating for Subject-Adaptive Riemannian Alignment in EEG Analysis},
  author  = {Tan, Huang and Li, Xiangzhu and Zhang, Li and Yin, Guangqiang},
  journal = {Journal of Neuroscience Methods},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Full bibliographic details will be updated upon publication.

---

## License

Released under the [MIT License](LICENSE).

---

## Acknowledgements

We thank the BCMI laboratory (Shanghai Jiao Tong University) and the DEAP team
(Queen Mary University of London) for making the SEED, SEED-IV, and DEAP datasets
publicly available.