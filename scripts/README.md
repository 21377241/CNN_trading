# scripts/ - 流水线执行脚本

## 文件说明

| 脚本 | 平台 | 用途 |
|------|------|------|
| `run_spx.sh` | Git Bash / WSL / Linux | SPX 美股完整流水线 |
| `run_spx.ps1` | PowerShell (Windows) | SPX 美股完整流水线 |
| `run_csi800.sh` | Git Bash / WSL / Linux | CSI800 A股完整流水线 |
| `run_csi800.ps1` | PowerShell (Windows) | CSI800 A股完整流水线 |
| `run_all.sh` | Git Bash / WSL / Linux | 同时运行两条流水线 |

---

## 快速上手

### PowerShell（Windows 推荐）

```powershell
# 首次需要设置执行策略（仅一次）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# SPX 完整运行
.\scripts\run_spx.ps1

# CSI800 完整运行
.\scripts\run_csi800.ps1
```

### Git Bash / WSL

```bash
# SPX 完整运行
bash scripts/run_spx.sh

# CSI800 完整运行
bash scripts/run_csi800.sh

# 同时运行两条流水线
bash scripts/run_all.sh
```

---

## 常用参数

### 断点续跑（跳过已完成步骤）

```powershell
# 从步骤 3 开始（已有预处理和图表数据）
.\scripts\run_spx.ps1 -FromStep 3
.\scripts\run_csi800.ps1 -FromStep 3
```

```bash
bash scripts/run_spx.sh    --from 3
bash scripts/run_csi800.sh --from 3
```

### 只运行指定步骤

```powershell
# 只重跑分析（步骤 4,5）
.\scripts\run_spx.ps1 -OnlySteps "4,5"
.\scripts\run_csi800.ps1 -OnlySteps "4"
```

```bash
bash scripts/run_spx.sh    --only 4,5
bash scripts/run_csi800.sh --only 4
```

### 预处理额外参数（CSI800 独有）

```powershell
# 只处理 2015-2025 年数据
.\scripts\run_csi800.ps1 -PreprocessArgs "--years 2015-2025"

# 强制只重算未来收益（步骤5）
.\scripts\run_csi800.ps1 -PreprocessArgs "--force-from 5"
```

```bash
bash scripts/run_csi800.sh --preprocess-args "--years 2015-2025"
bash scripts/run_csi800.sh --preprocess-args "--force-from 5"
```

### 空跑验证（不实际执行）

```powershell
.\scripts\run_spx.ps1    -DryRun
.\scripts\run_csi800.ps1 -DryRun
```

```bash
bash scripts/run_spx.sh    --dry-run
bash scripts/run_csi800.sh --dry-run
```

---

## 流水线步骤

### SPX（5步）

| 步骤 | 脚本 | 输出目录 |
|------|------|---------|
| 1 预处理 | `1_preprocess_spx_data.py` | `WORK_SPACE/data/processed_data/` |
| 2 图表生成 | `2_generate_spx_chart_data.py` | `WORK_SPACE/data/stocks_dataset/stocks_USA/` |
| 3 训练+OOS | `3_train_spx_model.py` | `WORK_SPACE/spx_model_res/` |
| 4 十分位分析 | `4_analyze_spx_portfolio.py` | `WORK_SPACE/spx_model_res/portfolio_simplified/` |
| 5 TopN分析 | `5_analyze_spx_topn_portfolio.py` | `WORK_SPACE/topN_portfolio/` |

### CSI800（4步）

| 步骤 | 脚本 | 输出目录 |
|------|------|---------|
| 1 预处理 | `1_preprocess_csi800_data.py` | `WORK_SPACE/data/csi800_processed_data/` |
| 2 图表生成 | `2_generate_csi800_chart_data.py` | `WORK_SPACE/data/csi800_stocks_dataset/stocks_CHN/` |
| 3 训练+OOS | `3_train_csi800_model.py` | `WORK_SPACE/csi800_model_res/` |
| 4 组合分析 | `4_analyze_csi800_portfolio.py` | `WORK_SPACE/csi800_model_res/portfolio_simplified/` + `topN_portfolio/` |

> CSI800 步骤4 同时输出**十分位分析**和 **TopN（5/10/20）分析**，无需单独步骤5。

---

## 日志

每次运行自动生成带时间戳的日志文件：

```
WORK_SPACE/logs/spx/run_20260331_183000.log
WORK_SPACE/logs/csi800/run_20260331_190000.log
```
