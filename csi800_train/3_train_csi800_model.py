# -*- coding: utf-8 -*-
"""
3_train_csi800_model.py - CSI800 CNN 模型训练与 OOS 预测

使用步骤 2 生成的图表数据训练 2D CNN 集成模型（5 个），
并对 CSI800_OOS_YEARS 进行预测，输出 ensem5_res_{year}.csv。

输出目录：
  WORK_SPACE/csi800_model_res/<model_name>/<exp_name>/
    checkpoint{0..4}.pth.tar
    ensem_res/ensem5_res_{year}.csv
"""
import sys
import os
import logging
import glob

# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

# ──────────────────────────────────────────────
from Misc.config import (
    CSI800_EXP_DIR, CSI800_LOG_DIR, CSI800_STOCKS_SAVEPATH,
    CSI800_IS_YEARS, CSI800_OOS_YEARS, CSI800_COUNTRY,
    BENCHMARK_MODEL_WINDOW, BENCHMARK_MODEL_PREDICT_WINDOW,
    LEARNING_RATE, DROPOUT_RATE, N_ENSEMBLES,
    BATCH_SIZE, MAX_EPOCHS, EARLY_STOP_PATIENCE,
    FREQ, MA_LAGS, VOLUME_BAR
)

# ──────────────────────────────────────────────
os.makedirs(CSI800_LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(CSI800_LOG_DIR, 'train_csi800.log'),
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
from Experiments.cnn_experiment import train_us_model
from Data.dgp_config import get_dataset_dir, get_dataset_filename


# CSI800 图表数据集目录（与步骤2生成时相同）
CSI800_DATASET_DIR = get_dataset_dir(CSI800_COUNTRY, base_savepath=CSI800_STOCKS_SAVEPATH)


def check_data_available(years: list) -> list:
    """返回有有效图表数据文件的年份列表"""
    available = []
    for year in years:
        fn_img = get_dataset_filename(
            BENCHMARK_MODEL_WINDOW, FREQ, VOLUME_BAR, MA_LAGS, year, '_images.dat')
        fn_lbl = get_dataset_filename(
            BENCHMARK_MODEL_WINDOW, FREQ, VOLUME_BAR, MA_LAGS, year, '_labels.feather')
        p_img = os.path.join(CSI800_DATASET_DIR, fn_img)
        p_lbl = os.path.join(CSI800_DATASET_DIR, fn_lbl)
        if os.path.exists(p_img) and os.path.exists(p_lbl):
            available.append(year)
        else:
            logger.warning(f"年份 {year} 的数据文件不存在，已从列表中移除。")
    return available


def main():
    logger.info("=" * 60)
    logger.info("  步骤 3（CSI800）：CNN 模型训练与 OOS 预测")
    logger.info("=" * 60)
    logger.info(f"训练年份  : {CSI800_IS_YEARS}")
    logger.info(f"测试年份  : {CSI800_OOS_YEARS}")
    logger.info(f"国家标识  : {CSI800_COUNTRY}")
    logger.info(f"实验目录  : {CSI800_EXP_DIR}")
    logger.info(f"数据目录  : {CSI800_DATASET_DIR}")
    logger.info(f"模型配置  : ws={BENCHMARK_MODEL_WINDOW}, freq={FREQ}")
    logger.info(f"训练配置  : lr={LEARNING_RATE}, dropout={DROPOUT_RATE}, "
                f"ensemble={N_ENSEMBLES}, batch={BATCH_SIZE}")

    # 检查数据可用性
    avail_is  = check_data_available(CSI800_IS_YEARS)
    avail_oos = check_data_available(CSI800_OOS_YEARS)

    if not avail_is:
        logger.error("样本内年份无有效数据！请先运行步骤 2（CSI800）。")
        logger.info(f"期望数据目录: {CSI800_DATASET_DIR}")
        return 1

    if not avail_oos:
        logger.warning("样本外年份无有效数据，将跳过 OOS 预测。")

    logger.info(f"实际训练年份: {avail_is}")
    logger.info(f"实际测试年份: {avail_oos}")

    # 启动训练（通过 base_exp_dir 和 dataset_dir 将结果写入 CSI800 专属目录）
    train_us_model(
        ws_list=[BENCHMARK_MODEL_WINDOW],
        pw_list=[BENCHMARK_MODEL_PREDICT_WINDOW],
        freq=FREQ,
        ma_lags=MA_LAGS,
        has_vb=VOLUME_BAR,
        lr=LEARNING_RATE,
        dropout=DROPOUT_RATE,
        n_ensembles=N_ENSEMBLES,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        is_years=avail_is,
        oos_years=avail_oos,
        train_val_split=0.7,
        country=CSI800_COUNTRY,
        base_exp_dir=CSI800_EXP_DIR,
        dataset_dir=CSI800_DATASET_DIR,
    )

    # 验证输出
    logger.info(f"\n{'='*60}")
    logger.info("验证输出结果：")
    ensem_pattern = os.path.join(CSI800_EXP_DIR, '**', 'ensem_res', '*.csv')
    ensem_files   = glob.glob(ensem_pattern, recursive=True)
    ckpt_pattern  = os.path.join(CSI800_EXP_DIR, '**', 'checkpoint*.pth.tar')
    ckpt_files    = glob.glob(ckpt_pattern, recursive=True)

    logger.info(f"  找到 checkpoint 文件: {len(ckpt_files)} 个")
    logger.info(f"  找到 OOS 预测文件: {len(ensem_files)} 个")
    for fp in sorted(ensem_files):
        import pandas as pd
        try:
            df = pd.read_csv(fp)
            logger.info(f"    {os.path.basename(fp)}: {len(df)} 条预测")
        except Exception:
            logger.info(f"    {os.path.basename(fp)}")

    logger.info("\n[OK] 步骤 3（CSI800）完成！")
    logger.info("  下一步请运行: python 4_analyze_csi800_portfolio.py")
    return 0


if __name__ == '__main__':
    sys.exit(main())
