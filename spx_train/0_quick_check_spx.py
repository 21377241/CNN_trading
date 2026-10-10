# -*- coding: utf-8 -*-
"""
0_quick_check_spx.py - 环境与数据检查

检查项：
  1. Python 版本 ≥ 3.7
  2. 必要依赖包
  3. CUDA / GPU 情况
  4. SPX_volume_price 目录与 CSV 文件列
  5. CACHE_DIR 下 SPY 文件（可选）
  6. 核心代码包文件
  7. 磁盘空间
  8. 工作目录（不存在则创建）
"""
import sys
import os
import shutil

# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

# ──────────────────────────────────────────────
PASS  = "  [OK]"
WARN  = "  [!] "
FAIL  = "  [XX]"
SEP   = "-" * 60


def check_python_version():
    print("\n=== 1. Python 版本 ===")
    v = sys.version_info
    if v >= (3, 7):
        print(f"{PASS} Python {v.major}.{v.minor}.{v.micro}")
    else:
        print(f"{FAIL} Python {v.major}.{v.minor}.{v.micro}（需要 >= 3.7）")
        return False
    return True


def check_packages():
    print("\n=== 2. 必要依赖包 ===")
    ok = True
    required = {
        'pandas':     'pandas',
        'numpy':      'numpy',
        'torch':      'torch',
        'PIL':        'Pillow',
        'tqdm':       'tqdm',
        'matplotlib': 'matplotlib',
        'pyarrow':    'pyarrow',
    }
    optional = ['chardet', 'torchsummary']

    for mod, pkg in required.items():
        try:
            __import__(mod)
            print(f"{PASS} {pkg}")
        except ImportError:
            print(f"{FAIL} {pkg}（请运行: pip install {pkg}）")
            ok = False

    for mod in optional:
        try:
            __import__(mod)
            print(f"{PASS} {mod}（可选）")
        except ImportError:
            print(f"{WARN} {mod} 未安装（可选，不影响主流程）")

    return ok


def check_cuda():
    print("\n=== 3. CUDA / GPU ===")
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            print(f"{PASS} CUDA 可用，设备数: {n}，GPU[0]: {name}")
        else:
            print(f"{WARN} CUDA 不可用，将使用 CPU 训练（速度较慢）")
    except Exception as e:
        print(f"{WARN} 检查 CUDA 时出错: {e}")
    return True  # 不强制要求 GPU


def check_data_dir():
    print("\n=== 4. SPX_volume_price 数据目录 ===")
    import glob
    import pandas as pd

    data_dir = os.path.join(PROJECT_ROOT, 'SPX_volume_price')
    if not os.path.isdir(data_dir):
        print(f"{FAIL} 目录不存在: {data_dir}")
        return False

    # 检查年份子目录
    year_dirs = sorted([
        d for d in glob.glob(os.path.join(data_dir, '*'))
        if os.path.isdir(d) and os.path.basename(d).isdigit()
    ])
    if not year_dirs:
        print(f"{FAIL} 未找到年份子目录（如 2011/, 2012/, ...）")
        return False

    years = [os.path.basename(d) for d in year_dirs]
    print(f"{PASS} 发现 {len(year_dirs)} 个年份目录: {years[0]} ~ {years[-1]}")

    # 统计总文件数（取前几年抽样）
    total_files = 0
    for yd in year_dirs[:3]:
        cnt = len(glob.glob(os.path.join(yd, '*.csv')))
        total_files += cnt
        print(f"       {os.path.basename(yd)}: {cnt} 个 CSV 文件")
    if len(year_dirs) > 3:
        print(f"       ...（共 {len(year_dirs)} 个年份）")

    if total_files == 0:
        print(f"{FAIL} 前几年目录下无 CSV 文件")
        return False

    # 抽样检查列
    required_cols = {'date', 'wind_code', 'OPEN', 'CLOSE', 'HIGH', 'LOW', 'VOLUME'}
    ok = True
    try:
        sample_dir = year_dirs[0]
        csv_files = sorted(glob.glob(os.path.join(sample_dir, '*.csv')))
        if csv_files:
            sample_file = csv_files[0]
            try:
                df = pd.read_csv(sample_file, encoding='gbk')
            except UnicodeDecodeError:
                df = pd.read_csv(sample_file, encoding='utf-8')

            actual_cols = set(df.columns)
            missing = required_cols - actual_cols
            if missing:
                print(f"{FAIL} 样本文件缺少列: {missing}")
                ok = False
            else:
                print(f"{PASS} 列检查通过（样本: {os.path.basename(sample_file)}）")
                print(f"       列名: {list(df.columns)}")
                print(f"       行数: {len(df)}，股票数: {df['wind_code'].nunique()}")
    except Exception as e:
        print(f"{WARN} 读取样本文件失败: {e}")

    return ok


def check_cache_dir():
    print("\n=== 5. CACHE_DIR 与 SPY 文件（可选） ===")
    cache_dir = os.path.join(PROJECT_ROOT, 'trend_submit', 'CACHE_DIR')
    if not os.path.isdir(cache_dir):
        print(f"{WARN} CACHE_DIR 不存在: {cache_dir}（可选，缺失时从数据自算期末日）")
        return True

    spy_files = ['spy_month_ret.csv', 'spy_week_ret.csv', 'spy_quarter_ret.csv']
    found_any = False
    for fn in spy_files:
        fp = os.path.join(cache_dir, fn)
        if os.path.exists(fp):
            print(f"{PASS} {fn}")
            found_any = True
        else:
            print(f"{WARN} {fn} 不存在（可选）")

    if not found_any:
        print(f"       → 将从 us_ret_spx.feather 自动推算周期末日期")

    return True


def check_code_package():
    print("\n=== 6. 核心代码包文件 ===")
    code_pkg = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
    files_to_check = [
        'Data/equity_data.py',
        'Data/generate_chart.py',
        'Data/chart_library.py',
        'Data/chart_dataset.py',
        'Data/dgp_config.py',
        'Model/cnn_model.py',
        'Experiments/cnn_experiment.py',
        'Misc/config.py',
        'Portfolio/portfolio.py',
    ]
    ok = True
    for fn in files_to_check:
        fp = os.path.join(code_pkg, fn)
        if os.path.exists(fp):
            print(f"{PASS} {fn}")
        else:
            print(f"{FAIL} {fn}（缺失）")
            ok = False
    return ok


def check_disk_space():
    print("\n=== 7. 磁盘空间 ===")
    try:
        total, used, free = shutil.disk_usage(PROJECT_ROOT)
        free_gb = free / (1024 ** 3)
        if free_gb >= 5:
            print(f"{PASS} 可用空间: {free_gb:.1f} GB")
        elif free_gb >= 1:
            print(f"{WARN} 可用空间: {free_gb:.1f} GB（建议 >= 5 GB）")
        else:
            print(f"{FAIL} 可用空间不足: {free_gb:.1f} GB")
            return False
    except Exception as e:
        print(f"{WARN} 磁盘空间检查失败: {e}")
    return True


def check_work_dirs():
    print("\n=== 8. 工作目录检查（不存在则创建）===")
    dirs = [
        os.path.join(PROJECT_ROOT, 'WORK_SPACE'),
        os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'data', 'processed_data'),
        os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'data', 'stocks_dataset',
                     'stocks_USA', 'dataset_all'),
        os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'spx_model_res'),
        os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'spx_model_res', 'log'),
        os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'spx_model_res', 'portfolio_simplified'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"{PASS} {os.path.relpath(d, PROJECT_ROOT)}")
    return True


# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("    CNN 股票收益预测项目 - 环境与数据检查")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")

    results = {
        'Python 版本':         check_python_version(),
        '依赖包':               check_packages(),
        'CUDA/GPU':            check_cuda(),
        'SPX_volume_price':   check_data_dir(),
        'CACHE_DIR':           check_cache_dir(),
        '代码包文件':           check_code_package(),
        '磁盘空间':             check_disk_space(),
        '工作目录':             check_work_dirs(),
    }

    print("\n" + SEP)
    print("检查结果汇总：")
    all_ok = True
    for name, passed in results.items():
        status = "通过" if passed else "失败"
        mark = PASS if passed else FAIL
        print(f"  {mark} {name}: {status}")
        if not passed:
            all_ok = False

    print(SEP)
    if all_ok:
        print("\n[OK] 所有检查通过！")
        print("  下一步请运行: python 1_preprocess_spx_data.py")
    else:
        print("\n[XX] 部分检查失败，请按提示修复后重试。")

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())

