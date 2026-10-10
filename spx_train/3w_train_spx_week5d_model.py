# -*- coding: utf-8 -*-
"""
3w_train_spx_week5d_model.py - SPX CNN 模型训练（20d输入 / 5d预测）

使用步骤 2w 生成的周频图表数据训练 2D CNN 集成模型（5 个），
并对 OOS_YEARS 进行预测，输出 ensem5_res_{year}.csv。

关键参数：
  - 输入窗口：20 天（图像尺寸 60×64，与月频相同）
  - 预测频率：week（每周末预测未来 5 个交易日涨跌）
  - 标签：Ret_week 的正负方向（binary_label）

输出目录：
  WORK_SPACE/spx_model_res/<model_name>/20d5p-lr1E-4-dp0.50-maTrue-vbTrue-weekTrained-noDelayedReturn/
    checkpoint{0..4}.pth.tar
    ensem_res/ensem5_res_{year}.csv
"""
import sys
import os
import logging

# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

# ──────────────────────────────────────────────
from Misc.config import (
    WORK_DIR,
    IS_YEARS, OOS_YEARS,
    BENCHMARK_MODEL_WINDOW,
    LEARNING_RATE, DROPOUT_RATE, N_ENSEMBLES,
    BATCH_SIZE, MAX_EPOCHS, EARLY_STOP_PATIENCE,
    MA_LAGS, VOLUME_BAR, COUNTRY
)

# ── 周频专属参数 ──
WEEK_FREQ = 'week'
WEEK_PREDICT_WINDOW = 5
WINDOW_SIZE = BENCHMARK_MODEL_WINDOW  # 20 天

# ── 新独立目录 ──
WEEK_EXP_DIR = os.path.join(WORK_DIR, 'spx_model_res_20d5p')
WEEK_LOG_DIR = os.path.join(WEEK_EXP_DIR, 'log')

# ──────────────────────────────────────────────
os.makedirs(WEEK_LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(WEEK_LOG_DIR, 'train_spx_week5d.log'),
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
from Experiments.cnn_experiment import train_us_model
from Data.dgp_config import get_dataset_dir, get_dataset_filename


def check_data_available(years: list) -> list:
    """返回有有效周频图表数据文件的年份列表"""
    available = []
    dataset_dir = get_dataset_dir(COUNTRY)
    for year in years:
        fn_img = get_dataset_filename(
            WINDOW_SIZE, WEEK_FREQ, VOLUME_BAR, MA_LAGS, year, '_images.dat')
        fn_lbl = get_dataset_filename(
            WINDOW_SIZE, WEEK_FREQ, VOLUME_BAR, MA_LAGS, year, '_labels.feather')
        p_img = os.path.join(dataset_dir, fn_img)
        p_lbl = os.path.join(dataset_dir, fn_lbl)
        if os.path.exists(p_img) and os.path.exists(p_lbl):
            available.append(year)
        else:
            logger.warning(f"年份 {year} 的周频数据文件不存在，已从列表中移除。")
    return available


def main():
    logger.info("=" * 60)
    logger.info("  步骤 3w：SPX CNN 训练（20d输入 / 5d预测）")
    logger.info("=" * 60)
    logger.info(f"训练年份    : {IS_YEARS}")
    logger.info(f"测试年份    : {OOS_YEARS}")
    logger.info(f"输入窗口    : {WINDOW_SIZE} 天")
    logger.info(f"预测频率    : {WEEK_FREQ}（{WEEK_PREDICT_WINDOW} 个交易日）")
    logger.info(f"训练配置    : lr={LEARNING_RATE}, dropout={DROPOUT_RATE}, "
                f"ensemble={N_ENSEMBLES}, batch={BATCH_SIZE}")

    avail_is  = check_data_available(IS_YEARS)
    avail_oos = check_data_available(OOS_YEARS)

    if not avail_is:
        logger.error("样本内年份无有效周频数据！请先运行步骤 2w。")
        return 1

    if not avail_oos:
        logger.warning("样本外年份无有效周频数据，将跳过 OOS 预测。")

    logger.info(f"实际训练年份: {avail_is}")
    logger.info(f"实际测试年份: {avail_oos}")

    train_us_model(
        ws_list=[WINDOW_SIZE],
        pw_list=[WEEK_PREDICT_WINDOW],
        freq=WEEK_FREQ,
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
        country=COUNTRY,
        base_exp_dir=WEEK_EXP_DIR,
    )

    # 验证输出
    logger.info(f"\n{'='*60}")
    logger.info("验证输出结果：")
    import glob
    ensem_pattern = os.path.join(WEEK_EXP_DIR, '**', 'ensem_res',
                                 f'ensem{N_ENSEMBLES}_res_*.csv')
    week_files = glob.glob(ensem_pattern, recursive=True)

    logger.info(f"  找到周频 OOS 预测文件: {len(week_files)} 个")
    for fp in sorted(week_files):
        import pandas as pd
        try:
            df = pd.read_csv(fp)
            logger.info(f"    {os.path.basename(fp)}: {len(df)} 条预测")
        except Exception:
            logger.info(f"    {os.path.basename(fp)}")

    logger.info(f"  模型目录: {WEEK_EXP_DIR}")
    logger.info("\n[OK] 步骤 3w 完成！下一步请运行: python 4w_analyze_spx_week5d_portfolio.py")
    return 0


if __name__ == '__main__':
    sys.exit(main())
