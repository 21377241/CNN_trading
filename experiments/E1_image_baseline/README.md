# E1 — Image baseline (I20/R5)

clean 分支只使用 CSMAR 数据（`docs/protocol/clean_spec.md` 第 9 节）。旧数据上的 E1/E1b 结果见 `docs/results/paper_results_E1b.md`（历史记录），其配置在 git 历史和 tag `clean-olddata-v1` 中。

## M2：干净版本 chart-CNN（clean_spec 5.1）

- 配置：[`config_cnn_hp_clean.yaml`](config_cnn_hp_clean.yaml)（原 E1b-ric 网格与三阶段流程；`training: clean`；OOS 2008–2026）
- 输入：`WORK_SPACE/data/ashare_stocks_dataset/I20R5_clean/`
- 产物：`WORK_SPACE/paper/E1c_cnn_hp_I20R5_clean/`

```powershell
python ashare_train/3b_cnn_hp_search.py --config experiments/E1_image_baseline/config_cnn_hp_clean.yaml --phase all
```

## 渲染器缺陷评估（clean_spec 第 6 节，只报告）

- M0：旧模型，预测在旧工作区（`config.PRE_CLEAN_WORK_DIR`）。
- M1：旧数据流程 + Jiang 渲染器。图像由 `ashare_train/2m1_rerender_legacy_windows.py` 生成，配置为 [`config_cnn_m1_rendfix.yaml`](config_cnn_m1_rendfix.yaml)。输入和产物都在 `WORK_SPACE/_olddata/`。
- 比较：[`eval_clean_cnn_impact.py`](eval_clean_cnn_impact.py) → `docs/results/clean_cnn_impact.md`
