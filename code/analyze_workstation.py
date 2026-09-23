"""汇总工作站 T1/T2/T3/T7/T8 最新结果 (20260715 批次)."""
import json
import os
import numpy as np
from scipy import stats

results_dir = 'results'

# ── 加载最新的 T1 文件 (每个数据集的最新文件) ──
# T1 是增量保存, 每个数据集/对齐器一个文件, 取最新的合并
def load_latest_t1():
    """加载最新的 T1 baseline 文件 (20260714_174001 之后的)."""
    # 工作站批次的时间戳从 20260714_174001 开始
    t1_files = [
        'd1_T1_baseline_20260714_174001.json',  # 第一个
        'd1_T1_baseline_20260714_181036.json',
        'd1_T1_baseline_20260714_184158.json',
        'd1_T1_baseline_20260714_211926.json',
        'd1_T1_baseline_20260714_233705.json',
        'd1_T1_baseline_20260714_235001.json',
        'd1_T1_baseline_20260714_235906.json',
        'd1_T1_baseline_20260715_004623.json',
        'd1_T1_baseline_20260715_013352.json',  # 最后一个
    ]
    # 合并所有文件 (后面的覆盖前面的同 dataset/aligner)
    merged = {}
    for fn in t1_files:
        fp = os.path.join(results_dir, fn)
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            data = json.load(f)
        for ds, ds_data in data.items():
            if ds not in merged:
                merged[ds] = {}
            for aligner, res in ds_data.items():
                merged[ds][aligner] = res
    return merged

print('=' * 90)
print('Workstation Results Summary (T1/T2/T3/T7/T8, 20260715 batch)')
print('=' * 90)

# ════════════════════════════════════════════════════════════════
# T1: 单数据集 LOSO Baseline
# ════════════════════════════════════════════════════════════════
print('\n' + '=' * 90)
print('T1: Single-Dataset LOSO Baseline')
print('=' * 90)
t1 = load_latest_t1()
aligners = ['none', 'euclidean', 'riemann', 'rpa']

for ds in ['SEED', 'SEED_IV', 'DEAP']:
    if ds not in t1:
        print(f'\n--- {ds}: NOT FOUND ---')
        continue
    print(f'\n--- {ds} ---')
    print('{:<12} {:<18} {:<18} {:<12}'.format('Aligner', 'ACC', 'F1', 'Time_min'))
    print('-' * 60)
    for a in aligners:
        if a not in t1[ds]:
            continue
        r = t1[ds][a]
        time_min = r.get('time_s', 0) / 60
        print('{:<12} {:.4f}+/-{:.4f}   {:.4f}+/-{:.4f}   {:.1f}'.format(
            a, r['acc_mean'], r['acc_std'], r['f1_mean'], r['f1_std'], time_min))

# T1 统计检验 (RPA vs none)
print('\n--- T1 Paired t-test: none vs others ---')
for ds in ['SEED', 'SEED_IV', 'DEAP']:
    if ds not in t1:
        continue
    print(f'\n  [{ds}]')
    for a in ['euclidean', 'riemann', 'rpa']:
        if a not in t1[ds] or 'none' not in t1[ds]:
            continue
        for metric in ['acc', 'f1']:
            va = np.array(t1[ds]['none'][f'{metric}_per_subject'])
            vb = np.array(t1[ds][a][f'{metric}_per_subject'])
            if len(va) != len(vb):
                continue
            t, p = stats.ttest_rel(vb, va)
            diff = (vb - va).mean()
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print('  none vs {:<10} {}: diff={:+.4f} p={:.4f} {}'.format(
                a, metric.upper(), diff, p, sig))

# ════════════════════════════════════════════════════════════════
# T2/T3: 跨数据集迁移
# ════════════════════════════════════════════════════════════════
print('\n' + '=' * 90)
print('T2/T3: Cross-Dataset Transfer')
print('=' * 90)

# T2
t2_path = os.path.join(results_dir, 'd1_T2_seed_to_deap_20260715_020712.json')
if os.path.exists(t2_path):
    with open(t2_path) as f:
        t2 = json.load(f)
    print('\n--- T2: SEED -> DEAP ---')
    print('{:<12} {:<18} {:<18}'.format('Aligner', 'ACC', 'F1'))
    print('-' * 50)
    for a in aligners:
        if a not in t2:
            continue
        r = t2[a]
        print('{:<12} {:.4f}+/-{:.4f}   {:.4f}+/-{:.4f}'.format(
            a, r['acc_mean'], r['acc_std'], r['f1_mean'], r['f1_std']))

# T3
t3_path = os.path.join(results_dir, 'd1_T3_deap_to_seed_20260715_023408.json')
if os.path.exists(t3_path):
    with open(t3_path) as f:
        t3 = json.load(f)
    print('\n--- T3: DEAP -> SEED ---')
    print('{:<12} {:<18} {:<18}'.format('Aligner', 'ACC', 'F1'))
    print('-' * 50)
    for a in aligners:
        if a not in t3:
            continue
        r = t3[a]
        print('{:<12} {:.4f}+/-{:.4f}   {:.4f}+/-{:.4f}'.format(
            a, r['acc_mean'], r['acc_std'], r['f1_mean'], r['f1_std']))

# ════════════════════════════════════════════════════════════════
# T7: Fréchet 距离 + Centering Index
# ════════════════════════════════════════════════════════════════
print('\n' + '=' * 90)
print('T7: Fréchet Distance + Centering Index')
print('=' * 90)

t7_path = os.path.join(results_dir, 'd1_T7_frechet_ci_20260715_025420.json')
if os.path.exists(t7_path):
    with open(t7_path) as f:
        t7 = json.load(f)

    # 跨数据集 Fréchet 距离
    print('\n--- Cross-Dataset Fréchet Distance (before alignment) ---')
    print('{:<25} {:<10} {:<10} {:<10} {:<10} {:<10}'.format(
        'Pair', '1-4Hz', '4-8Hz', '8-14Hz', '14-31Hz', '31-50Hz'))
    print('-' * 75)
    for pair, bands_data in t7.get('cross_dataset', {}).items():
        row = '{:<25}'.format(pair)
        for band in ['1-4Hz', '4-8Hz', '8-14Hz', '14-31Hz', '31-50Hz']:
            row += ' {:<10.3f}'.format(bands_data.get(band, 0))
        print(row)

    # Centering Index (CI < 1 = 对齐有效)
    print('\n--- Centering Index (CI < 1 means alignment effective) ---')
    for ds in ['SEED', 'SEED_IV', 'DEAP']:
        if ds not in t7.get('centering_index', {}):
            continue
        print(f'\n  [{ds}]')
        print('  {:<12} {:<10} {:<10} {:<10} {:<10} {:<10}'.format(
            'Aligner', '1-4Hz', '4-8Hz', '8-14Hz', '14-31Hz', '31-50Hz'))
        for a in ['euclidean', 'riemann', 'rpa']:
            if a not in t7['centering_index'][ds]:
                continue
            ci = t7['centering_index'][ds][a]
            row = '  {:<12}'.format(a)
            for band in ['1-4Hz', '4-8Hz', '8-14Hz', '14-31Hz', '31-50Hz']:
                val = ci.get(band, 0)
                row += ' {:<10.3f}'.format(val)
            print(row)

# ════════════════════════════════════════════════════════════════
# T8: SOTA 对比
# ════════════════════════════════════════════════════════════════
print('\n' + '=' * 90)
print('T8: SOTA Comparison')
print('=' * 90)

t8_path = os.path.join(results_dir, 'd1_T8_sota_comparison_20260715_032700.json')
if os.path.exists(t8_path):
    with open(t8_path) as f:
        t8 = json.load(f)

    for ds in ['SEED', 'SEED_IV', 'DEAP']:
        if ds not in t8:
            continue
        print(f'\n--- {ds} ---')
        print('{:<15} {:<18} {:<18} {:<15}'.format('Method', 'ACC', 'F1', 'Note'))
        print('-' * 70)
        for method, res in t8[ds].items():
            acc = res.get('acc_mean', 0)
            acc_std = res.get('acc_std', 0)
            f1 = res.get('f1_mean', 0)
            f1_std = res.get('f1_std', 0)
            note = res.get('note', res.get('method', ''))[:40]
            if f1 > 0:
                print('{:<15} {:.4f}+/-{:.4f}   {:.4f}+/-{:.4f}   {}'.format(
                    method, acc, acc_std, f1, f1_std, note))
            else:
                print('{:<15} {:.4f}+/-{:.4f}   {:<18}   {}'.format(
                    method, acc, acc_std, '-', note))

    # 跨数据集文献对比
    if '_cross_dataset_literature' in t8:
        print('\n--- Cross-Dataset Literature (SF-UDA) ---')
        cd_lit = t8['_cross_dataset_literature']
        for pair, methods in cd_lit.items():
            print(f'\n  {pair}:')
            for method, res in methods.items():
                print('    {}: ACC={:.4f} ({})'.format(
                    method, res.get('acc_mean', 0), res.get('note', '')[:60]))
