# E2 — Matched-Information Benchmark

对照设计见 [`docs/design/paper_e2_design.md`](../../docs/design/paper_e2_design.md)。旧数据上的结果见 [`docs/results/paper_results_E2.md`](../../docs/results/paper_results_E2.md)（历史记录）。

clean 分支只保留附录模型（HistGB、Logit、MLP-180）所需的训练和评估代码（`clean_spec.md` 第 8 节）。E2、E2A、E2B 的旧变体不保留，已删除，可在 git 历史中查到。

- `build_dataset.py`：由 universe 和 CSMAR 日面板单独重建 140 维特征（默认写到 `WORK_SPACE/paper/clean_inputs/features140`，与 `2_generate_ashare_chart_data.py` 的产物相同）。
- `run_train.py`、`run_eval.py`：`--config` 必填。
- `config_ind_chrono_ric.yaml`、`config_ind_chrono_ric_mlp4.yaml`：旧数据上的规范设定（标签、CNN 预测指向旧流程的目录，在 clean 工作区不存在）。C5 重跑附录模型时，另建读取 `clean_inputs` 和 registry v2 的配置。
