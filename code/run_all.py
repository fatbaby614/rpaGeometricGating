"""
run_all.py — 一键实验启动脚本
==============================

跨平台支持 (Windows / Linux), 自动适配路径。

用法:
    # 完整实验 (所有表)
    python run_all.py

    # 快速测试 (前 3 个被试, 仅 T1)
    python run_all.py --quick

    # 指定实验表
    python run_all.py --tables T1,T7

    # 仅跑特定数据集
    python run_all.py --tables T1 --datasets SEED

环境变量:
    RIEMANN_EMO_DATA_ROOT  数据集根目录 (Linux 推荐)
        例: export RIEMANN_EMO_DATA_ROOT=/mnt/data1/home/tanhuang/datasets
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from datetime import datetime


# ════════════════════════════════════════════════════════════════════
# 环境检查
# ════════════════════════════════════════════════════════════════════

def check_environment():
    """检查 Python 环境和依赖."""
    print(f"\n{'='*60}")
    print(f"Environment Check")
    print(f"{'='*60}")
    print(f"  Platform:  {platform.system()} {platform.release()}")
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  Working dir: {os.getcwd()}")
    
    # 检查数据根目录
    data_root = os.environ.get("RIEMANN_EMO_DATA_ROOT", "(not set)")
    print(f"  RIEMANN_EMO_DATA_ROOT: {data_root}")
    
    # 检查关键依赖
    missing = []
    for pkg in ["numpy", "scipy", "sklearn", "pyriemann"]:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (missing)")
            missing.append(pkg)
    
    if missing:
        print(f"\n[WARNING] 缺少依赖: {missing}")
        print(f"  安装: pip install {' '.join(missing)}")
        return False
    
    # 检查数据集路径
    print(f"\n  数据集路径检查:")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from paths import SEED_ROOT, SEED_IV_ROOT, DEAP_ROOT
        for name, path in [("SEED", SEED_ROOT), ("SEED_IV", SEED_IV_ROOT), ("DEAP", DEAP_ROOT)]:
            status = "✓" if path and os.path.isdir(path) else "✗"
            print(f"    {status} {name}: {path}")
    except Exception as e:
        print(f"    ✗ 路径检查失败: {e}")
        return False
    
    return True


# ════════════════════════════════════════════════════════════════════
# 实验执行
# ════════════════════════════════════════════════════════════════════

def run_experiment(script_path: str, args: list, description: str = ""):
    """运行一个实验脚本."""
    start_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'─'*60}")
    print(f"[{start_ts}] ▶ {description}")
    print(f"  Script: {script_path}")
    print(f"  Args: {' '.join(args)}")
    print(f"{'─'*60}", flush=True)

    cmd = [sys.executable, script_path] + args
    t0 = time.time()

    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        end_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{end_ts}] ✓ {description} 完成 ({elapsed:.1f}s)", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        end_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{end_ts}] ✗ {description} 失败 ({elapsed:.1f}s, exit code {e.returncode})", flush=True)
        return False


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RiemannianEmotionEEG 一键实验启动",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_all.py                          # 完整实验
  python run_all.py --quick                  # 快速测试 (前3被试)
  python run_all.py --tables T1,T7           # 仅 T1 和 T7
  python run_all.py --tables T1 --datasets SEED  # 仅 SEED 的 T1
        """
    )
    parser.add_argument("--tables", type=str, default="T1,T2,T3,T4,T4mc,T5,T6,T7,T8",
                        help="实验表, 逗号分隔 (T1,T2,T3,T4,T4mc,T5,T6,T7,T8,T9)")
    parser.add_argument("--datasets", type=str, default="SEED,SEED_IV,DEAP",
                        help="数据集, 逗号分隔")
    parser.add_argument("--aligners", type=str, default="none,euclidean,riemann,rpa",
                        help="对齐器, 逗号分隔")
    parser.add_argument("--channels", type=str, default="32ch",
                        help="通道配置 (32ch/16ch/8ch/4ch)")
    parser.add_argument("--bands", type=str, default="5band",
                        help="频段配置 (5band/8band)")
    parser.add_argument("--dl_baselines", type=str, default="",
                        help="T8 本地训练的 DL baseline, 逗号分隔 "
                             "(DGCNN-SPD, mdJPT). 空=不跑 DL baseline")
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式 (前 3 被试)")
    parser.add_argument("--skip_env_check", action="store_true",
                        help="跳过环境检查")
    args = parser.parse_args()
    
    # 时间戳
    start_time = datetime.now()
    print(f"\n{'#'*60}")
    print(f"# RiemannianEmotionEEG 一键实验")
    print(f"# 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    # 环境检查
    if not args.skip_env_check:
        if not check_environment():
            print("\n[ERROR] 环境检查失败, 请修复后重试")
            print("  或用 --skip_env_check 跳过检查")
            sys.exit(1)
    
    # 快速模式
    n_subjects = 3 if args.quick else None
    if args.quick:
        print(f"\n[QUICK MODE] 每数据集仅用前 {n_subjects} 个被试")
    
    # 实验脚本路径
    code_dir = os.path.dirname(os.path.abspath(__file__))
    d1_script = os.path.join(code_dir, "experiments", "run_d1_experiments.py")

    # 结果目录 (项目根/results/, code_dir=code/, 上溯一级即项目根)
    results_dir = os.path.join(os.path.dirname(code_dir), "results")
    print(f"\n[OUTPUT] 结果将保存到: {os.path.abspath(results_dir)}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 构建参数
    common_args = [
        "--datasets", args.datasets,
        "--aligners", args.aligners,
        "--channels", args.channels,
        "--bands", args.bands,
    ]
    if n_subjects:
        common_args.extend(["--n_subjects", str(n_subjects)])
    # --dl_baselines 透传 (T8 使用, 其他 table 忽略)
    if args.dl_baselines:
        common_args.extend(["--dl_baselines", args.dl_baselines])
    
    # 执行 D1 实验
    tables = args.tables.split(",")
    d1_args = common_args + ["--tables", ",".join(tables)]
    
    success = run_experiment(
        d1_script, d1_args,
        f"D1 Experiments (tables: {tables})"
    )
    
    # 汇总
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'#'*60}")
    print(f"# 实验汇总")
    print(f"#{'':>2}开始: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#{'':>2}结束: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#{'':>2}耗时: {duration:.1f}s ({duration/60:.1f}min)")
    print(f"#{'':>2}状态: {'✓ 成功' if success else '✗ 失败'}")
    print(f"#{'':>2}结果: {os.path.join(code_dir, '..', 'results')}")
    print(f"{'#'*60}")
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
