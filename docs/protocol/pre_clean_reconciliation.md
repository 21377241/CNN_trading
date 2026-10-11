# 新旧数据对账（只报告）

由 `ashare_train/reconcile_pre_clean.py` 生成。clean 分支的数据流程不读取任何旧数据；本文件只记录换源（旧数据商 → CSMAR）前后的差异，不作为任何构建、模型或结果的输入。详细结果见 `WORK_SPACE/data/ashare_processed_data/reconcile_pre_clean/<phase>.json`。

## panel（2026-10-11 19:39:15）

| 编号 | 检查 | 标准 | 结果 | 结论 |
|---|---|---|---|---|
| D1 | 与旧面板重叠股票-日的日收益 | 差异 ≤ 0.1 个百分点 ≥ 98%；超出的抽 100 条归因 | 重叠 15,946,387 行，98.3098% 在 0.1pp 内；超出 269,525 行；抽样归因 {'非除权日：旧面板复权价错误（旧涨跌幅与新一致）': 96, '除权日：复权方法差异': 4} | 符合预期 |
| D2 | 只在新或旧面板出现的股票-日 | 分类并解释 | 只在新面板 2,252,051：{'早于旧面板该股首日': 822410, '晚于旧面板截止日': 613086, '旧面板无此股票': 541560, '北交所': 274879, '其他': 116}；只在旧面板 14,574：{'已并入新代码的旧代码（旧面板在新代码下另有同日行，即重复历史）': 14561, 'CSMAR 标为停牌（Filling = 1）': 9, '已并入新代码的旧代码': 3, '其他': 1} | 仅报告 |
| F2 | BM、EP 与旧数据商 PB、PE 倒数的截面秩相关（月末，2008 起） | 中位数 ≥ 0.9 | 216 个月；BM 中位 0.997（最小 0.906）；EP 中位 0.984（最小 0.607） | 符合预期 |

## universe（2026-10-11 22:43:40）

```json
{
  "old_rows_all": 3329137,
  "old_rows_labeled_ewma_pos": 3110806,
  "new_rows": 3465838,
  "both": 3090480,
  "only_old": 20326,
  "only_old_no_trade_on_t": 2957,
  "only_old_st_on_t": 548,
  "only_new": 375358,
  "only_new_label_nan": 26949,
  "only_new_n_obs_lt_20": 8008,
  "label_absdiff_le_1e-9_share": 0.0007982578757992286,
  "label_absdiff_gt_1e-3_share": 0.024668983458880175,
  "oos_new_rows": 3034729,
  "oos_old_rows_all": 2857361
}
```

## inputs（2026-10-11 22:45:00）

```json
{
  "oos_keys_vs_m0": {
    "m0_rows": 2687822,
    "new_oos_rows": 3034729,
    "both": 2671619,
    "only_m0": 16203,
    "only_m0_ipo_rule": 14029,
    "only_m0_other": 2174,
    "only_new": 363110,
    "only_new_no_label": 18884,
    "only_new_with_missing_days_in_window": 5246,
    "only_new_share_of_new": 0.11965154054941973
  },
  "close_path_vs_pre_clean_vendor": {
    "windows": 34060,
    "share_max_dev_lt_1e-3": 0.9334409864944216,
    "median_max_dev": 0.00024246985406767285
  },
  "figure": "docs/figures/clean_render_compare.png"
}
```
