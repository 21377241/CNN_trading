# -*- coding: utf-8 -*-
"""
Misc/config.py - 全局配置文件
集中定义项目路径、训练参数等关键配置
"""
import os

# ========================
# 路径配置
# ========================
# 本文件位于 trend_submit/trend_code_submit/Misc/config.py
# 因此 PROJECT_ROOT 向上三层
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_PACKAGE_ROOT = os.path.dirname(_THIS_DIR)                        # trend_code_submit/
_TREND_SUBMIT = os.path.dirname(CODE_PACKAGE_ROOT)                    # trend_submit/
PROJECT_ROOT = os.path.dirname(_TREND_SUBMIT)                          # CNN_trading/

# 原始数据目录（SPX_volume_price，按年份分目录，每天一个 CSV）
SPX_DIR = os.path.join(PROJECT_ROOT, 'SPX_volume_price')

# 原始数据目录（CSI800_volume_price，按年份分目录，每天一个 CSV）
CSI800_DIR = os.path.join(PROJECT_ROOT, 'CSI800_volume_price')

# 工作空间：物理目录在 F 盘；E 盘项目下为目录联接（junction）→ 见 WORK_SPACE/README.md
# 可通过环境变量 CNN_TRADING_WORK_DIR 覆盖（绝对路径）
_DEFAULT_WORK_DIR = os.path.join(PROJECT_ROOT, 'WORK_SPACE')
WORK_DIR = os.environ.get('CNN_TRADING_WORK_DIR', _DEFAULT_WORK_DIR)

# 预处理后数据目录（SPX）
PROCESSED_DATA_DIR = os.path.join(WORK_DIR, 'data', 'processed_data')

# 预处理后数据目录（CSI800）
CSI800_PROCESSED_DATA_DIR = os.path.join(WORK_DIR, 'data', 'csi800_processed_data')

# 图表数据集目录（SPX）
STOCKS_SAVEPATH = os.path.join(WORK_DIR, 'data', 'stocks_dataset')

# 图表数据集目录（CSI800）
CSI800_STOCKS_SAVEPATH = os.path.join(WORK_DIR, 'data', 'csi800_stocks_dataset')

ASHARE_PROCESSED_DATA_DIR = os.path.join(WORK_DIR, 'data', 'ashare_processed_data')
ASHARE_STOCKS_SAVEPATH = os.path.join(WORK_DIR, 'data', 'ashare_stocks_dataset')

# CSMAR 原始数据（clean_spec 第 9 节“数据源更换为 CSMAR”）
CSMAR_ROOT = os.environ.get('CSMAR_ROOT', r'F:\csmar_ashare_daily')
SAMPLE_END = os.environ.get('CNN_TRADING_SAMPLE_END', '2026-09-30')
ASHARE_FUNDAMENTALS_PATH = os.path.join(ASHARE_PROCESSED_DATA_DIR, 'fundamentals_pit.parquet')
ASHARE_MARKET_PATH = os.path.join(ASHARE_PROCESSED_DATA_DIR, 'market_daily.parquet')

# 干净版本（clean-data）：统一日面板、周度样本键、模型输入
ASHARE_PANEL_PATH = os.path.join(ASHARE_PROCESSED_DATA_DIR, 'ashare_ret.feather')
ASHARE_STOCK_INFO_PATH = os.path.join(ASHARE_PROCESSED_DATA_DIR, 'ashare_stock_info.feather')
ASHARE_UNIVERSE_PATH = os.path.join(ASHARE_PROCESSED_DATA_DIR, 'universe_weekly.feather')
CLEAN_INPUTS_DIR = os.path.join(WORK_DIR, 'paper', 'clean_inputs')

# 旧数据（只读，不进入 clean 分支的任何构建、模型或结果）：只由 reconcile_pre_clean.py（新旧对账）和
# 渲染器缺陷评估（2m1_rerender_legacy_windows.py、eval_clean_cnn_impact.py 的 M0/M1）读取。
#   PRE_CLEAN_WORK_DIR：旧工作区（pre-clean-v1）：M0 预测、旧图像与旧面板
#   OLD_DATA_WORK_DIR：旧数据上的干净版本产物（tag clean-olddata-v1）与 M1
PRE_CLEAN_WORK_DIR = os.environ.get('CNN_TRADING_PRE_CLEAN_WORK_DIR', r'F:\LLM_trading\CNN_trading\WORK_SPACE')
OLD_DATA_WORK_DIR = os.path.join(WORK_DIR, '_olddata')

# 缓存目录（SPY 期末等）
CACHE_DIR = os.path.join(_TREND_SUBMIT, 'CACHE_DIR')

# ========================
# 实验输出目录（SPX）
# ========================
EXP_DIR = os.path.join(WORK_DIR, 'spx_model_res')
LOG_DIR = os.path.join(EXP_DIR, 'log')
LATEX_DIR = os.path.join(EXP_DIR, 'latex')
PORTFOLIO_DIR = os.path.join(EXP_DIR, 'portfolio_simplified')

# ========================
# 实验输出目录（CSI800）
# ========================
CSI800_EXP_DIR = os.path.join(WORK_DIR, 'csi800_model_res')
CSI800_LOG_DIR = os.path.join(CSI800_EXP_DIR, 'log')
CSI800_PORTFOLIO_DIR = os.path.join(CSI800_EXP_DIR, 'portfolio_simplified')

# ========================
# 年份配置（SPX：2011-2025）
# ========================
IS_YEARS = list(range(2011, 2020))      # 样本内训练年份（2011-2019）
OOS_YEARS = list(range(2020, 2026))     # 样本外测试年份（2020-2025）

# ========================
# 年份配置（CSI800：2007-2025，训练2007-2018，测试2019-2025）
# ========================
CSI800_IS_YEARS = list(range(2007, 2019))   # 样本内训练年份（2007-2018）
CSI800_OOS_YEARS = list(range(2019, 2026))  # 样本外测试年份（2019-2025）

# ========================
# A 股全市场：切分（训练 2000–2005、验证 2006–2007、OOS 2008-01 至 SAMPLE_END）见
# experiments/common/splits.py 与 ashare_train/build_universe.py
# ========================
ASHARE_COUNTRY = 'CHN'

# ========================
# 模型与训练超参数（对齐 Jiang et al. JF 2023）
# ========================
# 窗口与预测
BENCHMARK_MODEL_WINDOW = 20         # 输入窗口（天数）
BENCHMARK_MODEL_PREDICT_WINDOW = 20 # 预测窗口（天数，对应 month）

# CNN 架构名称（论文 20 日：末层 256 通道）
BENCHMARK_MODEL_NAME = 'D20L3F53S31D21MP21F53S11D11MP21F53S11D11MP21C256'

# 训练超参数（论文：Adam lr=1e-5，batch=128）
LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.50
N_ENSEMBLES = 5
BATCH_SIZE = 128
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 2

# 数据相关
FREQ = 'month'                 # 预测频率
MA_LAGS = [20]                 # 均线周期列表
VOLUME_BAR = True              # 是否包含成交量图

# 国家标识（用于文件命名）
COUNTRY = 'USA'
CSI800_COUNTRY = 'CHN'

# CSV 文件编码
SPX_CSV_ENCODING = 'gbk'
CSI800_CSV_ENCODING = 'gbk'

# CSI800 旧流程的日收益裁剪；A 股全市场干净版本不裁剪（clean_spec D1）
CSI800_RET_CLIP = 0.11

# 确保工作目录存在
for _d in [WORK_DIR, PROCESSED_DATA_DIR, CSI800_PROCESSED_DATA_DIR,
           ASHARE_PROCESSED_DATA_DIR, ASHARE_STOCKS_SAVEPATH,
           STOCKS_SAVEPATH, CSI800_STOCKS_SAVEPATH, CACHE_DIR,
           EXP_DIR, LOG_DIR, LATEX_DIR, PORTFOLIO_DIR,
           CSI800_EXP_DIR, CSI800_LOG_DIR, CSI800_PORTFOLIO_DIR]:
    os.makedirs(_d, exist_ok=True)

