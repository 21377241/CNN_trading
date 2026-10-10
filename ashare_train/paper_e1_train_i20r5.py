# -*- coding: utf-8 -*-
"""
paper_e1_train_i20r5.py — 论文 E1：chronological 切分下训练 I20/R5

Train 2005–2014 | Val 2015–2017 | Test 2018–2025
图表数据复用 ashare_stocks_dataset/I20R5/

用法:
  python ashare_train/paper_e1_train_i20r5.py
"""
import sys
import os
import glob
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import (
    ASHARE_STOCKS_SAVEPATH, ASHARE_COUNTRY, PAPER_LOG_DIR,
    PAPER_TRAIN_YEARS, PAPER_VAL_YEARS, PAPER_TEST_YEARS, PAPER_EXP_DIR_E1,
    LEARNING_RATE, DROPOUT_RATE, N_ENSEMBLES, BATCH_SIZE,
    MAX_EPOCHS, EARLY_STOP_PATIENCE, MA_LAGS, VOLUME_BAR
)
from Data.dgp_config import get_dataset_dir, get_dataset_filename
from Experiments.cnn_experiment import train_us_model

WS, PW, FREQ = 20, 5, 'week'


def _available(years, dataset_dir, logger):
    out = []
    for y in years:
        fn_i = get_dataset_filename(WS, FREQ, VOLUME_BAR, MA_LAGS, y, '_images.dat')
        fn_l = get_dataset_filename(WS, FREQ, VOLUME_BAR, MA_LAGS, y, '_labels.feather')
        if os.path.exists(os.path.join(dataset_dir, fn_i)) and \
           os.path.exists(os.path.join(dataset_dir, fn_l)):
            out.append(y)
        else:
            logger.warning(f"缺数据年份: {y}")
    return out


def main():
    os.makedirs(PAPER_LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(PAPER_LOG_DIR, 'e1_train_i20r5.log'),
                encoding='utf-8'
            )
        ]
    )
    logger = logging.getLogger(__name__)

    base_save = os.path.join(ASHARE_STOCKS_SAVEPATH, 'I20R5')
    dataset_dir = get_dataset_dir(ASHARE_COUNTRY, base_savepath=base_save)

    train_y = _available(PAPER_TRAIN_YEARS, dataset_dir, logger)
    val_y = _available(PAPER_VAL_YEARS, dataset_dir, logger)
    test_y = _available(PAPER_TEST_YEARS, dataset_dir, logger)

    logger.info("=" * 60)
    logger.info("  论文 E1 训练 I20/R5 (chronological)")
    logger.info(f"  Train={train_y}")
    logger.info(f"  Val  ={val_y}")
    logger.info(f"  Test ={test_y}")
    logger.info(f"  exp  ={PAPER_EXP_DIR_E1}")
    logger.info(f"  data ={dataset_dir}")
    logger.info("=" * 60)

    if not train_y or not val_y:
        logger.error("Train/Val 数据不足，请先生成 I20R5 图表")
        return 1
    if not test_y:
        logger.error("Test 数据不足")
        return 1

    train_us_model(
        ws_list=[WS],
        pw_list=[PW],
        freq=FREQ,
        ma_lags=MA_LAGS,
        has_vb=VOLUME_BAR,
        lr=LEARNING_RATE,
        dropout=DROPOUT_RATE,
        n_ensembles=N_ENSEMBLES,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        is_years=train_y,
        val_years=val_y,
        oos_years=test_y,
        country=ASHARE_COUNTRY,
        base_exp_dir=PAPER_EXP_DIR_E1,
        dataset_dir=dataset_dir,
    )

    ens = glob.glob(os.path.join(PAPER_EXP_DIR_E1, '**', 'ensem_res', '*.csv'), recursive=True)
    ckpt = glob.glob(os.path.join(PAPER_EXP_DIR_E1, '**', 'checkpoint*.pth.tar'), recursive=True)
    logger.info(f"[OK] checkpoints={len(ckpt)}  Test csv={len(ens)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
