# CNN 股票收益预测项目 — 实现方案与 Code Agent 复现指南

本文档面向 **Code Agent** 或人工开发者，用于从零复现本项目代码。按文档中的规格可实现完整的数据预处理 → 图表生成 → 模型训练 → 投资组合分析流水线。

---

## 重要前提：用 SPX_volume_price 替代无法获取的原始数据

- **原有项目** 使用的原始数据（如 CRSP 全市场日度数据、`us_920101-200731.csv` / `us_ret.feather` 等）**本项目无法获取**，因此**不依赖上述数据**。
- **本方案唯一必需的行情数据** 为 **SPX_volume_price**（标普500 成分股日度 OHLC + 成交量，按年/日 CSV 组织）。从该目录出发即可完成全流程：预处理 → 图表生成 → 训练 → 组合分析。
- **SPY 期末文件**（`CACHE_DIR/spy_month_ret.csv` 等）用于步骤 1 的周期末收益对齐；若**没有**这些文件，可按本文档 **5.4 节** 从 `us_ret_spx.feather` 的交易日历自算周期末日期，并完成步骤 1、2，**无需任何其他原始数据**。
- 复现时只需准备：**SPX_volume_price 目录** + 本仓库代码与配置；Code Agent 按本文档实现即可完成“用 SPX 替代原数据”的完整流水线。

---

## 1. 项目概述

### 1.1 目标

- **输入**：SPX（标普500）成分股日度行情（OHLC + 成交量），2011 年 4 月至 2025 年 12 月；**仅使用 SPX_volume_price，不依赖原有项目无法获取的原始数据**。
- **方法**：将价格序列转为 **K 线+成交量+均线** 的灰度图像，用 **2D CNN** 做二分类（涨/跌），集成 5 个模型取平均概率。
- **输出**：按预测概率分十分位，评估等权/市值加权组合及 H-L（做多最高分位、做空最低分位）策略表现。

### 1.2 技术栈

- **语言**：Python 3.7+
- **核心依赖**：pandas, numpy, torch, PIL (Pillow), tqdm, matplotlib
- **可选**：chardet（CSV 编码检测）, pyarrow（feather 读写）
- **运行环境**：建议带 CUDA 的 GPU（训练耗时约 8–24 小时）

### 1.3 顶层流水线（4 步）

```
步骤0（可选）: 0_quick_check_spx.py     — 环境与数据检查
步骤1: 1_preprocess_spx_data.py         — 原始 CSV → 标准化面板 feather
步骤2: 2_generate_spx_chart_data.py    — 面板 → 图表图像 + 标签（.dat + .feather）
步骤3: 3_train_spx_model.py            — 训练 2D CNN 集成，产出 checkpoint 与预测 CSV
步骤4: 4_analyze_spx_portfolio.py      — 预测结果 → 十分位收益、H-L、报告与图表
```

---

## 2. 目录与路径规范

复现时需统一约定根目录（如 `PROJECT_ROOT`），下列路径均基于该根目录。

### 2.1 目录结构

```
PROJECT_ROOT/
├── SPX_volume_price/                    # 原始数据（外部提供）
│   ├── readme.txt
│   ├── 2011/
│   │   ├── 20110401.csv
│   │   └── ...
│   ├── 2012/ ... 2025/
│   └── ...
├── trend_submit/
│   ├── CACHE_DIR/                       # 缓存：SPY 基准等
│   │   ├── spy_month_ret.csv
│   │   ├── spy_week_ret.csv
│   │   └── spy_quarter_ret.csv
│   └── trend_code_submit/               # 核心代码包
│       ├── Data/
│       │   ├── equity_data.py
│       │   ├── dgp_config.py
│       │   ├── generate_chart.py
│       │   ├── chart_library.py
│       │   └── chart_dataset.py
│       ├── Model/
│       │   └── cnn_model.py
│       ├── Experiments/
│       │   └── cnn_experiment.py
│       ├── Portfolio/
│       │   └── portfolio.py
│       └── Misc/
│           ├── config.py
│           ├── utilities.py
│           └── cache_manager.py (若存在)
├── WORK_SPACE/
│   ├── data/
│   │   ├── processed_data/
│   │   │   └── us_ret_spx.feather      # 步骤1 输出
│   │   └── stocks_dataset/
│   │       └── stocks_USA/
│   │           └── dataset_all/        # 步骤2 输出
│   │               ├── 20d_month_has_vb_[20]_ma_YYYY_images.dat
│   │               ├── 20d_month_has_vb_[20]_ma_YYYY_labels.feather
│   │               └── 20d_month_has_vb_[20]_ma_YYYY.txt
│   └── spx_model_res/                  # 步骤3 输出
│       ├── D20L3F53S31D21MP21F53S11D11MP21F53S11D11MP21C64/
│       │   └── 20d20p-lr1E-04-dp0.50-maTrue-vbTrue-monthlyTrained-noDelayedReturn/
│       │       ├── checkpoint0.pth.tar ... checkpoint4.pth.tar
│       │       ├── ensem_res/
│       │       │   ├── ensem5_res_2019.csv ... ensem5_res_2025.csv
│       │       └── ...
│       ├── log/
│       ├── latex/
│       └── portfolio_simplified/       # 步骤4 输出
│           ├── portfolio_summary.txt
│           ├── ew_portfolio.csv
│           ├── vw_portfolio.csv
│           └── 1_decile_returns.png, 2_sharpe_ratios.png, ...
├── docs/
│   ├── SPX_volume_price_数据结构说明.md
│   └── 项目实现方案_CodeAgent复现指南.md  # 本文档
├── 0_quick_check_spx.py
├── 1_preprocess_spx_data.py
├── 2_generate_spx_chart_data.py
├── 3_train_spx_model.py
└── 4_analyze_spx_portfolio.py
```

### 2.2 关键路径变量（建议集中配置）

| 变量名 | 含义 | 示例 |
|--------|------|------|
| `SPX_DIR` | 原始 SPX CSV 根目录 | `PROJECT_ROOT/SPX_volume_price` |
| `WORK_DIR` | 工作空间根目录 | `PROJECT_ROOT/WORK_SPACE` |
| `PROCESSED_DATA_DIR` | 预处理后数据目录 | `WORK_DIR/data/processed_data` |
| `STOCKS_SAVEPATH` | 图表数据集根目录 | `WORK_DIR/data/stocks_dataset` |
| `CACHE_DIR` | 缓存目录（SPY 等） | `PROJECT_ROOT/trend_submit/CACHE_DIR` |
| `CODE_PACKAGE_ROOT` | 核心代码包根目录 | `PROJECT_ROOT/trend_submit/trend_code_submit` |

---

## 3. 原始数据规格（SPX_volume_price）

- **目录**：按年分目录 `2011/` … `2025/`，其下为单日 CSV，文件名 `YYYYMMDD.csv`。
- **单文件 CSV 列**（必须存在）：`date`, `wind_code`, `OPEN`, `CLOSE`, `HIGH`, `LOW`, `VOLUME`；可选 `sec_name`。
- **编码**：可能是 UTF-8 / GBK / GB18030 等，读取时需多编码尝试或 chardet 检测。
- 详细字段与约束见：`docs/repro/SPX_volume_price_数据结构说明.md`。

---

## 4. 步骤 0：环境与数据检查（0_quick_check_spx.py）

### 4.1 功能

- 检查 Python 版本 ≥ 3.7、必需包、CUDA/GPU、SPX 目录与抽样 CSV 列、CACHE_DIR 下 SPY 文件、代码包关键文件、磁盘空间、工作目录是否存在（不存在则创建）。

### 4.2 实现要点

- 必需包：`pandas`, `numpy`, `torch`, `PIL`（pillow）, `tqdm`, `matplotlib`。
- SPX：遍历 `SPX_DIR` 下年份子目录，统计 `*.csv` 数量，抽样一个 CSV 检查是否包含 `date`, `wind_code`, `OPEN`, `CLOSE`, `HIGH`, `LOW`, `VOLUME`。
- CACHE_DIR：若有 `spy_month_ret.csv`（及可选 `spy_week_ret.csv`, `spy_quarter_ret.csv`），步骤 1 用其日期做周期末收益对齐；**若没有 SPY 文件**，按 **5.4 节** 从 SPX 数据自算期末日即可，步骤 0 对 SPY 可做可选检查（缺失时提示，不阻止后续执行）。
- 代码包：检查 `Data/equity_data.py`, `Data/generate_chart.py`, `Data/chart_library.py`, `Data/chart_dataset.py`, `Data/dgp_config.py`, `Model/cnn_model.py`, `Experiments/cnn_experiment.py`, `Misc/config.py`, `Portfolio/portfolio.py` 是否存在。
- 输出：打印每项检查结果，全部通过时提示下一步执行 `1_preprocess_spx_data.py`。

---

## 5. 步骤 1：数据预处理（1_preprocess_spx_data.py）

### 5.1 输入与输出

- **输入**：`SPX_DIR` 下各年目录中的全部 `YYYYMMDD.csv`。
- **输出**：单文件 `PROCESSED_DATA_DIR/us_ret_spx.feather`，约 300–500MB。

### 5.2 列名标准化

将原始列名映射为：

- `date` → `Date`
- `wind_code` → `StockID`
- `OPEN` → `Open`, `CLOSE` → `Close`, `HIGH` → `High`, `LOW` → `Low`, `VOLUME` → `Vol`

仅保留与后续计算相关的列；`sec_name` 可不写入 feather。

### 5.3 处理流程（顺序执行）

1. **读取与合并**  
   按年遍历 CSV，使用多编码尝试或 chardet 解码，合并为单一 DataFrame。`Date` 转为 `datetime64`，按 `[StockID, Date]` 排序。

2. **日收益率**  
   `Ret = groupby(StockID)['Close'].pct_change()`，首日 Ret 置 0。

3. **市值代理**  
   可选方案：  
   - `volume_proxy`：`MarketCap = Vol * Close`，再按日 median 缩放至约 10 亿量级；  
   - `constant`：用最近一年截面 `Vol*Close` 的股票均值映射到全样本；  
   - `external`：从外部 CSV 按 `(Date, StockID)` 合并 `MarketCap`。

4. **EWMA 波动率**  
   `EWMA_vol = groupby(StockID)['Ret'].transform(lambda x: (x**2).ewm(alpha=0.05, adjust=False).mean().shift(1))`，缺失填 0。

5. **未来收益**  
   - 对数收益：`log_ret = log(1+Ret)`, `cum_log_ret = groupby(StockID)['log_ret'].cumsum()`。  
   - 固定 horizon：对 `days in [5,20,60,65,180,250,260]`，  
     `Ret_{days}d = exp(cum_log_ret.shift(-days) - cum_log_ret) - 1`。  
   - 周期末收益：若存在 `CACHE_DIR/spy_{freq}_ret.csv`（freq in week/month/quarter），取其 `date` 列为周期末日，仅在这些日上计算下一周期收益，再 merge 回主表，列名 `Ret_week`/`Ret_month`/`Ret_quarter`。  
   - 年度：以每年最后一个交易日为年末日，同样用 `cum_log_ret` 算下一年收益，列名 `Ret_year`。

6. **写出**  
   列中需包含：`Date`, `StockID`, `Open`, `High`, `Low`, `Close`, `Vol`, `Ret`, `MarketCap`, `EWMA_vol`, `log_ret`, `cum_log_ret`, `Ret_5d` … `Ret_260d`, `Ret_week`/`Ret_month`/`Ret_quarter`（若有）, `Ret_year`。  
   可增加 `Shares = MarketCap/Close` 便于后续。  
   以 `Date`、`StockID` 为索引或列均可，保存为 feather（需 pyarrow）。

### 5.4 期末日期的备用逻辑（无 SPY 时）

若缺少 `spy_*_ret.csv`，可从 feather 的 `Date` 去重后按周期取最后一个交易日：  
- week：按 (year, week) 取 max(Date)；  
- month：按 (year, month) 取 max(Date)；  
- quarter：按 (year, quarter) 取 max(Date)；  
- year：按 year 取 max(Date)。  
步骤 2 的“期末日”可完全由此计算，不依赖 SPY 文件。

---

## 6. 步骤 2：图表数据生成（2_generate_spx_chart_data.py + Data 层）

### 6.1 数据流

- **输入**：`us_ret_spx.feather`（步骤 1 输出）、`CACHE_DIR` 下 SPY 期末表（或从 feather 自算期末日）。
- **输出**（按年）：  
  - `{STOCKS_SAVEPATH}/stocks_USA/dataset_all/20d_month_has_vb_[20]_ma_{YEAR}_images.dat`  
  - 同前缀 `_labels.feather`  
  - 同前缀 `.txt`（日志：样本数、缺失统计）

### 6.2 与 Data 层的对接方式

步骤 2 的入口脚本应：

- 将 `sys.path` 加入 `trend_submit/trend_code_submit`，以便 `from Data.generate_chart import GenerateStockData` 等。
- 通过 **Monkey Patch** 替换数据源与期末日：
  - `equity_data.processed_US_data` → 新函数：读取 `us_ret_spx.feather`，设索引 `[Date, StockID]`，返回 DataFrame。
  - `equity_data.get_period_end_dates(period)` → 新函数：从 `us_ret_spx.feather` 的 `Date` 按 period（week/month/quarter/year）计算期末日序列并返回。

这样 `GenerateStockData` 内部调用的 `get_processed_US_data_by_year` / `get_period_end_dates` 都会走 SPX 数据与 SPX 期末日。

### 6.3 GenerateStockData 行为（Data/generate_chart.py）

- **构造参数**：country, year, window_size（如 20）, freq（如 "month"）, chart_freq=1, ma_lags（如 [20]）, volume_bar=True/False, need_adjust_price=True, chart_type="bar"。
- **数据**：调用 `equity_data.get_processed_US_data_by_year(year)`，得到当年及前若干年数据（用于 lookback）。
- **对每只股票、每个期末日**（由 `get_period_end_dates(freq)` 过滤出该 year）：
  - 取该日之前 `window_size` 个交易日 OHLCV + 已算好的 Ret、EWMA_vol、MarketCap 等。
  - 若需要均线，在窗口内计算 `Close` 的 `rolling(ma_lag).mean()`，列名如 `ma20`。
  - **价格归一化**：以窗口内第一天收盘为 1，按收益率递推调整 Open/High/Low/Close（见下方 adjust_price 逻辑）。
  - 调用 **chart_library** 绘制一张图：K 线（OHLC 柱状）+ 可选成交量柱 + 可选均线，输出灰度图，尺寸由 `dgp_config` 的 `IMAGE_WIDTH[chart_len]`、`IMAGE_HEIGHT[chart_len]` 决定（如 20 日 → 60×64）。
  - 图像转成 uint8 一维数组，与当日的 Ret、Ret_5d/20d/…、Ret_month、MarketCap、EWMA_vol、StockID、Date 等一起写入一条样本。
- **保存**：  
  - 图像：所有样本的 uint8 数组堆叠为二维 `(N, H*W)`，用 `np.memmap` 写入 `*_images.dat`。  
  - 标签与元数据：写入 `*_labels.feather`（不含 image 列）。  
  - 日志：样本数、各缺失类型计数写入 `*.txt`。

### 6.4 价格归一化（adjust_price）

- 窗口内第一天收盘设为 1.0，当日 Open/High/Low 按比例缩放。
- 对后续日：用 `Ret` 递推 `Close = (1+Ret)*prev_Close`，再按当日 Close 与原始 OHLC 比例缩放当日 Open/High/Low。
- 保证窗口内第一天 Close=1，便于跨股票可比。

### 6.5 绘图规格（Data/chart_library.py）

- **DrawOHLC** 输入：已归一化的 OHLC DataFrame，可选 vol 列、ma 列列表，chart_type in ["bar","pixel","centered_pixel"]。
- **K 线**：每根 bar 宽度由 `dgp_config.BAR_WIDTH`（如 3 像素），High-Low 竖线，Open/Close 横线；颜色与背景由 `BACKGROUND_COLOR`、`CHART_COLOR` 控制（如 0 与 255）。
- **成交量**：若 has_volume_bar，在图像底部预留高度，按日成交量比例画竖线。
- **均线**：在 OHLC 图层上画折线，使用同一 CHART_COLOR。
- 图像竖直方向：y 轴向上为价格高，保存前可 `transpose(Image.FLIP_TOP_BOTTOM)` 使与常见坐标系一致。
- 输出：PIL Image 模式 "L"，尺寸 `(width, height)`，如 (60, 64)。

### 6.6 配置（Data/dgp_config.py）

- 从 `Misc.config` 取 `WORK_DIR`，派生 `DATA_DIR`、`PROCESSED_DATA_DIR`、`STOCKS_SAVEPATH`。
- 定义 `BAR_WIDTH`, `IMAGE_WIDTH`, `IMAGE_HEIGHT`（按 5/20/60 日不同），`VOLUME_CHART_GAP`，`BACKGROUND_COLOR`，`CHART_COLOR`，`FREQ_DICT`（5→week, 20→month, 60→quarter 等）。
- `CACHE_DIR` 指向 SPY 等缓存目录。

### 6.7 1D CNN 时间序列数据（可选）

- `GenerateStockData.save_annual_ts_data()`：从同一窗口提取 OHLC+Vol+MA 等 6 维时间序列，形状 (N, 6, window_size)，存为 npz 或类似，供 1D CNN 使用。步骤 2 入口脚本可逐年调用 `save_annual_data()` 与 `save_annual_ts_data()`。

---

## 7. 步骤 3：模型训练（3_train_spx_model.py + Experiments + Model）

### 7.1 输入与输出

- **输入**：步骤 2 生成的 `*_images.dat`、`*_labels.feather`（按年），以及 `Misc.config` 中的 `IS_YEARS`（如 2012–2018）、`OOS_YEARS`（如 2019–2025）。
- **输出**：  
  - `WORK_DIR/spx_model_res/<model_name>/<exp_name>/checkpoint{0..4}.pth.tar`  
  - 同目录下 `ensem_res/ensem5_res_{year}.csv`（各测试年一份）  
  - `log/`、`latex/` 等（由 Experiments 写入）。

### 7.2 配置覆盖（入口脚本内）

- 在导入 `Experiments.cnn_experiment` 之前，修改 `Misc.config`：  
  - `EXP_DIR`、`PORTFOLIO_DIR`、`LOG_DIR`、`LATEX_DIR` 指向 `WORK_DIR/spx_model_res` 下子目录。  
  - `IS_YEARS` = [2012..2018]，`OOS_YEARS` = [2019..2025]（与步骤 2 的 SPX 年份一致）。

### 7.3 模型架构（Model/cnn_model.py）

- **名称约定**：如 `D20L3F53S31D21MP21F53S11D11MP21F53S11D11MP21C64` 表示 20 日输入、3 层卷积、filter/stride/dilation/maxpool 序列、最终 64 通道、分类数 2。
- **输入**：2D CNN 为 (batch, 1, H, W)，如 (B, 1, 64, 60)；1D CNN 为 (B, 6, T)。
- **结构**：若干 Conv2d（或 Conv1d）+ BatchNorm + LeakyReLU + MaxPool + Dropout，最后 Flatten + Linear → 2 类 logits。
- **输出**：2 类（涨/跌），用 softmax 得到 up_prob。

### 7.4 训练逻辑（Experiments/cnn_experiment.py）

- **train_us_model(ws_list, pw_list, ...)**：  
  - 对 `(ws, pw)` 组合（如 ws=20, pw=20）和每个 ensemble 下标（如 0..4）：  
    - 用 `EquityDataset`（或 1D 的 `TS1DDataset`）加载 `IS_YEARS` 对应年的 images+labels，按 year 合并后按 0.7/0.3 划分 train/val（或按时间划分）。  
    - 初始化 `cnn_model.Model` 与优化器（如 Adam lr=1e-4），训练循环：前向 → 交叉熵损失 → 反向 → 更新；在验证集上算 MCC/accuracy，early stop（如 2 轮 val loss 不降则停）。  
    - 保存 checkpoint（含 `model_state_dict`）到 `checkpoint{i}.pth.tar`。  
  - 对每个 OOS 年，加载 5 个 checkpoint 做预测，对 up_prob 做平均，写入 `ensem5_res_{year}.csv`，列至少包含：StockID、ending_date、up_prob、down_prob、ret_val、MarketCap 等。

### 7.5 数据集（Data/chart_dataset.py）

- **EquityDataset**：  
  - 根据 year、window_size、predict_window、freq、has_volume_bar、has_ma 等，构造文件名（与步骤 2 命名一致），读取对应 `*_images.dat`（memmap）和 `*_labels.feather`。  
  - 标签列：使用与 predict_window 对应的收益列（如 Ret_month）二值化为 0/1（或 -99 表示缺失），过滤掉 label=-99 或 EWMA_vol==0 的样本。  
  - `__getitem__` 返回 (image_tensor, label)，image 归一化方式可由 ts_scale 控制（如 image_scale 即 /255 或减均值除标准差）。

### 7.6 评估指标

- 训练/验证：MCC（Matthews Correlation Coefficient）、Accuracy、Cross-Entropy Loss。  
- 早停以验证集 loss 或 MCC 为准。

---

## 8. 步骤 4：投资组合分析（4_analyze_spx_portfolio.py）

### 8.1 输入与输出

- **输入**：`ensem5_res_{year}.csv`（2019–2025），需包含列：`ending_date`, `StockID`, `up_prob`, `ret_val`, `MarketCap`。
- **输出**：  
  - `portfolio_simplified/portfolio_summary.txt`  
  - `ew_portfolio.csv`、`vw_portfolio.csv`（十分位 + H-L 行的月度/年化收益、标准差、Sharpe）  
  - 图：`1_decile_returns.png`、`2_sharpe_ratios.png`、`3_hl_comparison.png`、`4_monthly_returns_timeseries.png`。

### 8.2 逻辑

- 合并各年 `ensem5_res_*.csv`，将 `ending_date` 转为 datetime。
- 按 `up_prob` 做十分位分组（`pd.qcut(up_prob, 10, labels=1..10)`）。
- **等权**：每个十分位内、每月，对 `ret_val` 求平均，得到每月收益序列；再算均值、标准差、年化收益、Sharpe（年化收益/年化波动，年化因子 sqrt(12)）。
- **市值加权**：同月同十分位内，以 `MarketCap` 为权对 `ret_val` 加权平均，再算同样统计量。
- **H-L**：Decile 10 收益减 Decile 1 收益（等权或市值加权），同样算月度序列、均值、年化、Sharpe。
- 报告与图表：十分位收益柱状图、Sharpe 柱状图、H-L 对比、月度收益时间序列（如 D1/D5/D10 与 H-L）。

---

## 9. 依赖与配置汇总

### 9.1 最小依赖

```
pandas
numpy
torch
pillow
tqdm
matplotlib
pyarrow   # feather
```

可选：`chardet`（CSV 编码）、`torchsummary`（模型摘要）。

### 9.2 关键配置项（复现时需统一）

| 位置 | 配置项 | 含义 | 典型值 |
|------|--------|------|--------|
| 步骤1 | START_YEAR, END_YEAR | 处理年份范围 | 2011, 2025 |
| 步骤1 | MARKET_CAP_METHOD | 市值估算 | volume_proxy |
| 步骤2 | WINDOW_SIZE, FREQ | 窗口与预测频率 | 20, "month" |
| 步骤2 | MA_LAGS, VOLUME_BAR | 均线周期、是否要成交量 | [20], True |
| 步骤3 | IS_YEARS, OOS_YEARS | 训练/测试年 | 2012–2018, 2019–2025 |
| 步骤3 | lr, ensem, dp | 学习率、集成数、Dropout | 1e-4, 5, 0.50 |
| 步骤4 | TEST_YEARS | 参与组合分析的年份 | 2019..2025 |

### 9.3 路径占位

复现时应将脚本中的绝对路径（如 `E:\phd\LLM_trading\cnn\...`）改为基于环境变量或配置文件的 `PROJECT_ROOT`，保证不同机器可运行。

---

## 10. Code Agent 复现检查清单

按下列顺序实现并自检，即可复现完整流水线（**仅依赖 SPX_volume_price，不依赖原 CRSP/us_ret 等无法获取的数据**）：

- [ ] **路径与配置**：在 `Misc.config`、`Data.dgp_config` 中集中定义 PROJECT_ROOT、WORK_DIR、PROCESSED_DATA_DIR、STOCKS_SAVEPATH、CACHE_DIR；入口脚本可覆盖 IS_YEARS/OOS_YEARS、EXP_DIR 等。
- [ ] **步骤 0**：实现 0_quick_check_spx.py，检查 Python、包、GPU、**SPX_volume_price 目录与列**、CACHE_DIR（SPY 可选）、代码包文件、磁盘与目录。
- [ ] **步骤 1**：实现 1_preprocess_spx_data.py，包含多编码读取、列名映射、Ret/MarketCap/EWMA_vol/未来收益计算、feather 写出；若无 SPY，实现从 Date 推导期末日并计算 Ret_week/month/quarter/year。
- [ ] **Data 层**：实现 equity_data（processed_US_data、get_period_end_dates、get_processed_US_data_by_year）、dgp_config、chart_library（DrawOHLC：K 线+量+均线）、generate_chart（GenerateStockData：load_adjusted_daily_prices、adjust_price、_generate_daily_features、save_annual_data、save_annual_ts_data）、chart_dataset（EquityDataset、TS1DDataset）。
- [ ] **步骤 2**：实现 2_generate_spx_chart_data.py，Monkey Patch 数据源与期末日，按年调用 GenerateStockData，校验生成的 .dat、.feather、.txt。
- [ ] **Model 层**：实现 cnn_model（Model、CNNModel/CNN1DModel、get_full_model_name、init_model），输入尺寸与 config 中 BENCHMARK_MODEL_* 一致。
- [ ] **Experiments 层**：实现 cnn_experiment（Experiment、train_us_model）：DataLoader 构建、训练循环、早停、checkpoint 保存、OOS 预测与 ensem5_res_*.csv 写出；可选 Portfolio 计算。
- [ ] **步骤 3**：实现 3_train_spx_model.py，设置 SPX 的 EXP_DIR/IS_YEARS/OOS_YEARS，调用 train_us_model(ws_list=[20], pw_list=[20], ...)。
- [ ] **步骤 4**：实现 4_analyze_spx_portfolio.py，读 ensem5_res_*.csv、十分位分组、等权/市值加权/H-L 统计、报告与 4 张图。

---

## 11. 文档与参考

- 原始数据结构：`docs/repro/SPX_volume_price_数据结构说明.md`  
- 项目概览与快速开始：根目录 `SPX_README.md`（若存在）  
- 本文档：`docs/repro/项目实现方案_CodeAgent复现指南.md`

按本实现方案，Code Agent 可从零还原数据预处理、图表生成、CNN 训练与投资组合分析全流程；**全流程仅依赖 SPX_volume_price 数据，不依赖原有项目中的 CRSP/us_ret 等无法获取的原始数据**。实现时请保持与上述路径、列名、文件名及配置一致，以便与现有脚本和结果兼容。
