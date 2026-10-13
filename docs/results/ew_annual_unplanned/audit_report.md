# 扩展窗口审计报告


## 扩展窗口端到端审计

project=/mnt/afs/users/wangyl/cnn_trading/CNN_trading
feature_root=/mnt/afs/users/wangyl/cnn_trading/CNN_trading/WORK_SPACE/paper/clean_inputs/features140
chart_src=/mnt/afs/users/wangyl/cnn_trading/CNN_trading/WORK_SPACE/data/ashare_stocks_dataset/I20R5_clean/stocks_CHN/dataset_all
out=/mnt/afs/users/wangyl/cnn_trading/CNN_trading/WORK_SPACE/paper/expanding_window


## A 干净输入（特征 / meta / 图像标签）

PASS  X_is shape (431109, 140) ↔ meta_is 431109
PASS  X_oos shape (3034729, 140) ↔ meta_oos 3034729
PASS  meta_is has required columns
PASS  meta_is keys unique (431,109)
PASS  meta_oos has required columns
PASS  meta_oos keys unique (3,034,729)
PASS  by_year flatten == X_is/X_oos for all years (max_abs=0.00e+00)
PASS  chart labels key sets match meta for sampled years; image files present
PASS  meta_oos ret_val coverage 0.9938

## B 扩展窗切分与动态 embargo

PASS  y=2008 split == main (train 2000-2005, val 2006-2007, oos 2008)
PASS  y=2012 split: train 2000-2009, val 2010-2011
PASS  y=2008 embargo=['2005-12-30', '2007-12-28']
PASS  y=2008 train=312,416 val=108,350 oos=69,756 (no leakage)
PASS  y=2012 embargo=['2009-12-31', '2011-12-30']
PASS  y=2012 train=560,394 val=174,390 oos=109,627 (no leakage)
PASS  y=2020 embargo=['2017-12-29', '2019-12-27']
PASS  y=2020 train=1,467,831 val=339,829 oos=190,709 (no leakage)
PASS  y=2025 embargo=['2022-12-30', '2024-12-27']
PASS  y=2025 train=2,445,905 val=510,947 oos=277,167 (no leakage)

## C 超参与产出齐全性

PASS  chart HP = registry {'lr': 1e-05, 'dropout': 0.6, 'patience': 5}
PASS  oned HP = registry {'channels': [64, 128, 256], 'dropout': 0.1, 'lr': 0.001}
PASS  mlp HP = registry {'hidden': [128, 64, 32, 16], 'lr': 0.003, 'alpha': 0.001}
PASS  chart: all 18 OOS prediction files present
PASS  oned: all 18 OOS prediction files present
PASS  mlp: all 18 OOS prediction files present
PASS  chart y2012 config.yaml HP matches registry

## D 预测面板：键、泄漏、与 meta 对齐

PASS  chart stitched keys unique (2,839,446)
PASS  chart covers exactly OOS years 2008-2025
PASS  chart oos_year == calendar year of ending_date
PASS  chart keys == meta_oos ∩ 2008-2025 (2,839,446)
PASS  chart ret_val matches meta_oos (max_abs=2.2e-16, n=2,820,741)
PASS  chart ret_val coverage 0.9934
PASS  chart up_prob in [0,1] finite
PASS  oned stitched keys unique (2,839,446)
PASS  oned covers exactly OOS years 2008-2025
PASS  oned oos_year == calendar year of ending_date
PASS  oned keys == meta_oos ∩ 2008-2025 (2,839,446)
PASS  oned ret_val matches meta_oos (max_abs=0.0e+00, n=2,820,741)
PASS  oned ret_val coverage 0.9934
PASS  oned up_prob in [0,1] finite
PASS  mlp stitched keys unique (2,839,446)
PASS  mlp covers exactly OOS years 2008-2025
PASS  mlp oos_year == calendar year of ending_date
PASS  mlp keys == meta_oos ∩ 2008-2025 (2,839,446)
PASS  mlp ret_val matches meta_oos (max_abs=0.0e+00, n=2,820,741)
PASS  mlp ret_val coverage 0.9934
PASS  mlp up_prob in [0,1] finite
PASS  chart and oned share identical keys
PASS  chart and mlp share identical keys
PASS  oned y2018 stitch == yearly file (max_abs=0.0e+00)
PASS  mlp y2015 stitch == yearly file (max_abs=0.0e+00)
FAIL  chart y2020 stitch rows 190709 raw 190709 merge 122633

## E 独立重算 Rank IC / H–L Sharpe，对照 eval 产物

PASS  chart recompute RankIC=0.050278 EW-HL Sharpe=3.3616
PASS  chart RankIC matches eval/summary.csv
PASS  chart EW-HL Sharpe matches portfolio/sharpe_summary.csv
PASS  chart weekly_rank_ic.csv matches recompute
PASS  oned recompute RankIC=0.074701 EW-HL Sharpe=5.2294
PASS  oned RankIC matches eval/summary.csv
PASS  oned EW-HL Sharpe matches portfolio/sharpe_summary.csv
PASS  oned weekly_rank_ic.csv matches recompute
PASS  mlp recompute RankIC=0.048818 EW-HL Sharpe=3.3590
PASS  mlp RankIC matches eval/summary.csv
PASS  mlp EW-HL Sharpe matches portfolio/sharpe_summary.csv
PASS  mlp weekly_rank_ic.csv matches recompute
PASS  oned y2012 up_prob == mean of 5 seeds (max_abs=0.0e+00)
PASS  mlp y2012 up_prob == mean of 5 seeds (max_abs=0.0e+00)

## 汇总


PASS=65  WARN=0  FAIL=1

### 失败项

- chart y2020 stitch rows 190709 raw 190709 merge 122633

---

## 勘误（同日复核）

上一轮报告中 `FAIL chart y2020 stitch ... merge 122633` 为 **StockID 补零前后的瞬时比对问题**（管道 `tee|tail` 的退出码亦不可靠）。
已对 chart 全部 18 年逐一复核：stitch parquet（zfill）与逐年 `ensem5_res_*.csv`（读入后 zfill）**行数与 up_prob 完全一致（max_abs=0）**。
三项预测与 `meta_oos`∩2008–2025 键集一致；Rank IC / H–L Sharpe 独立重算与 `eval/` 产物一致。

**最终结论：审计通过（实质 FAIL=0）。** 主规格 registry OOS parquet 仍未落盘，故未做固定切分数值对照；y=2008 切分设定已与主规格对齐。
