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

# 工作空间
WORK_DIR = os.path.join(PROJECT_ROOT, 'WORK_SPACE')

# 预处理后数据目录
PROCESSED_DATA_DIR = os.path.join(WORK_DIR, 'data', 'processed_data')

# 图表数据集目录
STOCKS_SAVEPATH = os.path.join(WORK_DIR, 'data', 'stocks_dataset')

# 缓存目录（SPY 期末等）
CACHE_DIR = os.path.join(_TREND_SUBMIT, 'CACHE_DIR')

# ========================
# 实验输出目录
# ========================
EXP_DIR = os.path.join(WORK_DIR, 'spx_model_res')
LOG_DIR = os.path.join(EXP_DIR, 'log')
LATEX_DIR = os.path.join(EXP_DIR, 'latex')
PORTFOLIO_DIR = os.path.join(EXP_DIR, 'portfolio_simplified')

# ========================
# 年份配置（根据 SPX_volume_price 的时间范围，2011-2025）
# ========================
IS_YEARS = list(range(2011, 2020))      # 样本内训练年份（2011-2019）
OOS_YEARS = list(range(2020, 2026))     # 样本外测试年份（2020-2025）

# ========================
# 模型与训练超参数
# ========================
# 窗口与预测
BENCHMARK_MODEL_WINDOW = 20         # 输入窗口（天数）
BENCHMARK_MODEL_PREDICT_WINDOW = 20 # 预测窗口（天数，对应 month）

# CNN 架构名称（遵循文档约定）
BENCHMARK_MODEL_NAME = 'D20L3F53S31D21MP21F53S11D11MP21F53S11D11MP21C64'

# 训练超参数
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.50
N_ENSEMBLES = 5
BATCH_SIZE = 64
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 2

# 数据相关
FREQ = 'month'                 # 预测频率
MA_LAGS = [20]                 # 均线周期列表
VOLUME_BAR = True              # 是否包含成交量图

# 国家标识（用于文件命名）
COUNTRY = 'USA'

# CSV 文件编码（SPX_volume_price 使用 GBK 编码）
SPX_CSV_ENCODING = 'gbk'

# 确保工作目录存在
for _d in [WORK_DIR, PROCESSED_DATA_DIR, STOCKS_SAVEPATH, CACHE_DIR,
           EXP_DIR, LOG_DIR, LATEX_DIR, PORTFOLIO_DIR]:
    os.makedirs(_d, exist_ok=True)

