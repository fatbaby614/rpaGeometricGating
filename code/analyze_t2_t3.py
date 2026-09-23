"""T2/T3 跨数据集迁移结果汇总 + 统计检验 + 文献对比."""
import json
import numpy as np
from scipy import stats

# 读取 T2 和 T3 结果
with open('results/d1_T2_seed_to_deap_20260714_165440.json') as f:
    t2 = json.load(f)
with open('results/d1_T3_deap_to_seed_20260714_171900.json') as f:
    t3 = json.load(f)

aligners = ['none', 'euclidean', 'riemann', 'rpa']

# SF-UDA 文献数值 (Imtiaz 2026, 32ch 公共子集, 二分类)
sf_uda = {
    'SEED_to_DEAP': 0.6138,
    'DEAP_to_SEED': 0.6956,
}

print('=' * 90)
print('T2/T3 Cross-Dataset Transfer Results (32ch, 5band, SVM, max_ep=100)')
print('=' * 90)

# T2: SEED -> DEAP
print('\n--- T2: SEED -> DEAP (source=45 sessions, target=32 subjects) ---')
print('{:<12} {:<18} {:<18} {:<15}'.format('Aligner', 'ACC', 'F1', 'vs SF-UDA ACC'))
print('-' * 65)
for a in aligners:
    r = t2[a]
    delta = r['acc_mean'] - sf_uda['SEED_to_DEAP']
    print('{:<12} {:.4f}+/-{:.4f}   {:.4f}+/-{:.4f}   {:+.4f}'.format(
        a, r['acc_mean'], r['acc_std'], r['f1_mean'], r['f1_std'], delta))
print('SF-UDA (lit)  {:.4f}              -                  (baseline)'.format(
    sf_uda['SEED_to_DEAP']))

# T3: DEAP -> SEED
print('\n--- T3: DEAP -> SEED (source=32 subjects, target=45 sessions) ---')
print('{:<12} {:<18} {:<18} {:<15}'.format('Aligner', 'ACC', 'F1', 'vs SF-UDA ACC'))
print('-' * 65)
for a in aligners:
    r = t3[a]
    delta = r['acc_mean'] - sf_uda['DEAP_to_SEED']
    print('{:<12} {:.4f}+/-{:.4f}   {:.4f}+/-{:.4f}   {:+.4f}'.format(
        a, r['acc_mean'], r['acc_std'], r['f1_mean'], r['f1_std'], delta))
print('SF-UDA (lit)  {:.4f}              -                  (baseline)'.format(
    sf_uda['DEAP_to_SEED']))

# 统计检验
print('\n' + '=' * 90)
print('Paired t-test (T2: SEED->DEAP)')
print('=' * 90)
pairs = [('none', 'euclidean'), ('none', 'riemann'), ('none', 'rpa'),
         ('euclidean', 'riemann'), ('euclidean', 'rpa'), ('riemann', 'rpa')]
for a, b in pairs:
    for metric in ['acc', 'f1']:
        va = np.array(t2[a][f'{metric}_per_subject'])
        vb = np.array(t2[b][f'{metric}_per_subject'])
        t, p = stats.ttest_rel(vb, va)
        diff = (vb - va).mean()
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print('{:>10} vs {:<10} {}: diff={:+.4f} p={:.4f} {}'.format(
            a, b, metric.upper(), diff, p, sig))
    print()

print('=' * 90)
print('Paired t-test (T3: DEAP->SEED)')
print('=' * 90)
for a, b in pairs:
    for metric in ['acc', 'f1']:
        va = np.array(t3[a][f'{metric}_per_subject'])
        vb = np.array(t3[b][f'{metric}_per_subject'])
        t, p = stats.ttest_rel(vb, va)
        diff = (vb - va).mean()
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print('{:>10} vs {:<10} {}: diff={:+.4f} p={:.4f} {}'.format(
            a, b, metric.upper(), diff, p, sig))
    print()

# 方向不对称性分析
print('=' * 90)
print('Directional Asymmetry (SEED->DEAP vs DEAP->SEED)')
print('=' * 90)
print('{:<12} {:<15} {:<15} {:<15}'.format('Aligner', 'F1 (S->D)', 'F1 (D->S)', 'diff'))
print('-' * 60)
for a in aligners:
    f1_sd = t2[a]['f1_mean']
    f1_ds = t3[a]['f1_mean']
    print('{:<12} {:.4f}          {:.4f}          {:+.4f}'.format(
        a, f1_sd, f1_ds, f1_ds - f1_sd))

# 混淆矩阵
print('\n' + '=' * 90)
print('Confusion Matrices')
print('=' * 90)
labels = ['neg(0)', 'pos(1)']
for table_name, table_data in [('T2 (SEED->DEAP)', t2), ('T3 (DEAP->SEED)', t3)]:
    print('\n--- {} ---'.format(table_name))
    for a in aligners:
        cm = np.array(table_data[a]['confusion_matrix'])
        total = cm.sum()
        print('\n  {}:'.format(a.upper()))
        print('  {:<10} {:<10} {:<10} {:<10}'.format('', 'pred_neg', 'pred_pos', 'recall'))
        for i, lab in enumerate(labels):
            recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            print('  true_{:<7} {:<10} {:<10} {:.3f}'.format(lab, cm[i, 0], cm[i, 1], recall))
        acc = np.trace(cm) / total
        print('  Overall acc (CM): {:.4f}'.format(acc))

# 关键发现
print('\n' + '=' * 90)
print('Key Findings')
print('=' * 90)
print()
print('1. T2 (SEED->DEAP):')
print('   - Best: euclidean (F1=0.4521), worst: none (F1=0.3060)')
print('   - riemann/rpa F1~0.35, WORSE than euclidean')
print('   - SF-UDA (lit) ACC=0.6138, our best euclidean ACC=0.4851 (gap=-0.13)')
print()
print('2. T3 (DEAP->SEED):')
print('   - Best: riemann (F1=0.4282), worst: none (F1=0.3946)')
print('   - rpa F1=0.4164, close to riemann')
print('   - SF-UDA (lit) ACC=0.6956, our best riemann ACC=0.4304 (gap=-0.27)')
print()
print('3. Directional asymmetry:')
for a in aligners:
    f1_sd = t2[a]['f1_mean']
    f1_ds = t3[a]['f1_mean']
    better = 'D->S' if f1_ds > f1_sd else 'S->D'
    print('   {:<12}: S->D={:.4f} vs D->S={:.4f}  ({})'.format(a, f1_sd, f1_ds, better))
