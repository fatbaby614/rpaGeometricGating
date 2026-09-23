"""T1 结果汇总 + 统计检验 (临时分析脚本)."""
import json
import numpy as np
from scipy import stats

with open('results/d1_T1_baseline_20260714_151453.json') as f:
    data = json.load(f)['DEAP']

aligners = ['none', 'euclidean', 'riemann', 'rpa']
results = {}
for a in aligners:
    results[a] = {
        'acc': np.array(data[a]['acc_per_subject']),
        'f1': np.array(data[a]['f1_per_subject']),
        'acc_mean': data[a]['acc_mean'],
        'f1_mean': data[a]['f1_mean'],
        'acc_std': data[a]['acc_std'],
        'f1_std': data[a]['f1_std'],
        'time_s': data[a]['time_s'],
    }

print('=' * 80)
print('DEAP T1 Baseline Results (32 subjects, LOSO, 5band, SVM)')
print('=' * 80)
header = '{:<12} {:<18} {:<18} {:<10}'.format('Aligner', 'ACC', 'F1', 'Time_min')
print(header)
print('-' * 60)
for a in aligners:
    r = results[a]
    row = '{:<12} {:.4f}+/-{:.4f}   {:.4f}+/-{:.4f}   {:.1f}'.format(
        a, r['acc_mean'], r['acc_std'], r['f1_mean'], r['f1_std'], r['time_s'] / 60)
    print(row)

print()
print('=' * 80)
print('Paired t-test + Permutation test (10000 iterations)')
print('=' * 80)
pairs = [('none', 'euclidean'), ('none', 'riemann'), ('none', 'rpa'),
         ('euclidean', 'riemann'), ('euclidean', 'rpa'), ('riemann', 'rpa')]
for a, b in pairs:
    for metric in ['acc', 'f1']:
        d = results[b][metric] - results[a][metric]
        t, p = stats.ttest_rel(results[b][metric], results[a][metric])
        diff = d.mean()
        # permutation test
        rng = np.random.default_rng(42)
        n = len(d)
        count = 0
        for _ in range(10000):
            signs = rng.choice([-1, 1], size=n)
            if abs((d * signs).mean()) >= abs(diff):
                count += 1
        p_perm = count / 10000
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print('{:>10} vs {:<10} {}: diff={:+.4f} t={:+.2f} p_ttest={:.4f} {}  p_perm={:.4f}'.format(
            a, b, metric.upper(), diff, t, p, sig, p_perm))
    print()

print('=' * 80)
print('Key findings')
print('=' * 80)
f1_none = results['none']['f1_mean']
f1_riem = results['riemann']['f1_mean']
f1_rpa = results['rpa']['f1_mean']
f1_eucl = results['euclidean']['f1_mean']
print('F1 improvement (none -> euclidean): {:+.4f} ({:+.1f}%)'.format(
    f1_eucl - f1_none, (f1_eucl - f1_none) / f1_none * 100))
print('F1 improvement (none -> riemann): {:+.4f} ({:+.1f}%)'.format(
    f1_riem - f1_none, (f1_riem - f1_none) / f1_none * 100))
print('F1 improvement (none -> rpa): {:+.4f} ({:+.1f}%)'.format(
    f1_rpa - f1_none, (f1_rpa - f1_none) / f1_none * 100))
print('F1 improvement (euclidean -> rpa): {:+.4f}'.format(f1_rpa - f1_eucl))
print('F1 improvement (riemann -> rpa): {:+.4f}'.format(f1_rpa - f1_riem))

# Confusion matrix analysis
print()
print('=' * 80)
print('Confusion Matrices (aggregated across all 32 LOSO folds)')
print('=' * 80)
labels = ['neg(0)', 'pos(1)']
for a in aligners:
    cm = np.array(data[a]['confusion_matrix'])
    total = cm.sum()
    print('\n{}:'.format(a.upper()))
    print('  {:<10} {:<10} {:<10} {:<10}'.format('', 'pred_neg', 'pred_pos', 'recall'))
    for i, lab in enumerate(labels):
        recall = cm[i, i] / cm[i].sum()
        print('  true_{:<7} {:<10} {:<10} {:.3f}'.format(lab, cm[i, 0], cm[i, 1], recall))
    acc = np.trace(cm) / total
    print('  Overall acc (from CM): {:.4f}'.format(acc))
