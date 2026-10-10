# -*- coding: utf-8 -*-
"""3_train_ashare_model.py - A股全市场 CNN 训练（论文对齐超参）

用法:
  python ashare_train/3_train_ashare_model.py --spec I20R20
  python ashare_train/3_train_ashare_model.py --spec I5R5
  python ashare_train/3_train_ashare_model.py --spec I20R5 --lr 1e-5 --dropout 0.5
"""
import sys
import os
import glob
import argparse
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import (
    ASHARE_STOCKS_SAVEPATH, ASHARE_IS_YEARS, ASHARE_OOS_YEARS, ASHARE_COUNTRY,
    ASHARE_EXP_DIR_I20, ASHARE_EXP_DIR_I5, ASHARE_EXP_DIR_I20R5, ASHARE_LOG_DIR,
    LEARNING_RATE, DROPOUT_RATE, N_ENSEMBLES, BATCH_SIZE,
    MAX_EPOCHS, EARLY_STOP_PATIENCE, MA_LAGS, VOLUME_BAR
)
from Data.dgp_config import get_dataset_dir, get_dataset_filename
from Experiments.cnn_experiment import train_us_model

SPECS = {
    'I20R20': dict(
        window_size=20, predict_window=20, freq='month',
        exp_dir=ASHARE_EXP_DIR_I20,
    ),
    'I5R5': dict(
        window_size=5, predict_window=5, freq='week',
        exp_dir=ASHARE_EXP_DIR_I5,
    ),
    'I20R5': dict(
        window_size=20, predict_window=5, freq='week',
        exp_dir=ASHARE_EXP_DIR_I20R5,
    ),
}


def _parse_years(s: str | None) -> list | None:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(',')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', choices=list(SPECS.keys()) + ['all'], default='I20R20')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None, help='early_stop_patience')
    parser.add_argument('--n-ensembles', type=int, default=None)
    parser.add_argument('--out-root', type=str, default=None, help='覆盖 base_exp_dir')
    parser.add_argument('--train-years', type=str, default=None, help='逗号分隔，chrono train')
    parser.add_argument('--val-years', type=str, default=None, help='逗号分隔，chrono val')
    parser.add_argument('--dir-suffix', type=str, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--skip-oos', action='store_true')
    args = parser.parse_args()
    spec = SPECS[args.spec]
    ws, pw, freq = spec['window_size'], spec['predict_window'], spec['freq']
    exp_dir = args.out_root or spec['exp_dir']

    os.makedirs(ASHARE_LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(ASHARE_LOG_DIR, f'train_{args.spec}.log'),
                encoding='utf-8'
            )
        ]
    )
    logger = logging.getLogger(__name__)

    base_save = os.path.join(ASHARE_STOCKS_SAVEPATH, args.spec)
    dataset_dir = get_dataset_dir(ASHARE_COUNTRY, base_savepath=base_save)

    def available(years):
        out = []
        for y in years:
            fn_i = get_dataset_filename(ws, freq, VOLUME_BAR, MA_LAGS, y, '_images.dat')
            fn_l = get_dataset_filename(ws, freq, VOLUME_BAR, MA_LAGS, y, '_labels.feather')
            if os.path.exists(os.path.join(dataset_dir, fn_i)) and \
               os.path.exists(os.path.join(dataset_dir, fn_l)):
                out.append(y)
            else:
                logger.warning(f"缺数据年份: {y}")
        return out

    lr = args.lr if args.lr is not None else LEARNING_RATE
    dropout = args.dropout if args.dropout is not None else DROPOUT_RATE
    n_ensem = args.n_ensembles if args.n_ensembles is not None else N_ENSEMBLES
    patience = args.patience if args.patience is not None else EARLY_STOP_PATIENCE
    val_years = _parse_years(args.val_years)
    train_years = _parse_years(args.train_years)

    if train_years:
        is_y = available(train_years)
    else:
        is_y = available(ASHARE_IS_YEARS)
    oos_y = available(ASHARE_OOS_YEARS)
    if val_years:
        val_y = available(val_years)
    else:
        val_y = None

    logger.info("=" * 60)
    logger.info(f"  训练 [{args.spec}]  ws={ws} pw={pw} freq={freq}")
    logger.info(f"  lr={lr} dropout={dropout} patience={patience} ensemble={n_ensem}")
    logger.info(f"  IS={is_y}  val={val_y}  OOS={ASHARE_OOS_YEARS}")
    logger.info(f"  exp={exp_dir}")
    logger.info(f"  data={dataset_dir}")
    logger.info("=" * 60)

    if not is_y:
        logger.error("无训练数据，请先生成图表")
        return 1

    train_us_model(
        ws_list=[ws],
        pw_list=[pw],
        freq=freq,
        ma_lags=MA_LAGS,
        has_vb=VOLUME_BAR,
        lr=lr,
        dropout=dropout,
        n_ensembles=n_ensem,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        early_stop_patience=patience,
        is_years=is_y,
        oos_years=oos_y,
        val_years=val_y,
        train_val_split=0.7,
        country=ASHARE_COUNTRY,
        base_exp_dir=exp_dir,
        dataset_dir=dataset_dir,
        dir_suffix=args.dir_suffix,
        skip_oos=args.skip_oos,
        force_retrain=args.force,
    )

    ens = glob.glob(os.path.join(exp_dir, '**', 'ensem_res', '*.csv'), recursive=True)
    ckpt = glob.glob(os.path.join(exp_dir, '**', 'checkpoint*.pth.tar'), recursive=True)
    logger.info(f"[OK] checkpoints={len(ckpt)}  OOS csv={len(ens)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
