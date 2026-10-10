# E2G 结果：参数量匹配的 Chart-CNN 容量控制

设计见 `docs/design/paper_E2G_chart_capacity_design.md`。代码：`experiments/E2G_chart_capacity/run_e2g.py`（训练）、`eval_e2g.py`（评估）。输出：`WORK_SPACE/paper/E2G_chart_capacity/`，报告 `results/e2g_report.md`。

## 1. 研究问题

原 Jiang-style I20 chart-CNN（64-128-256，708,866 参数）在 A 股样本中弱于 1D-CNN-120（7,073 参数）与 MLP-4-120（26,369 参数）。这一差距是否主要来自可训练参数量的失配？

## 2. 实验方案

- 只改变三个卷积 block 的宽度，严格按 `[k, 2k, 4k]` 比例缩放；k 由参数量公式 `P(k) = 150k² + 1476k + 2` 机械选取，不参考任何 validation/OOS 结果。
- ChartCNN-S：k=4，4-8-16，8,306 参数（对标 1D-CNN-120）。
- ChartCNN-M：k=9，9-18-36，25,436 参数（对标 MLP-4-120）。
- 其余全部固定：renderer、`1×64×60` 输入、20 日窗口、未来五日二分类标签、三层深度、`5×3` kernel、第一层 stride/dilation、BN、LeakyReLU、`2×1` max pooling、flatten head。
- 切分：2000–2005 训练，2006–2007 validation，2008–2025 OOS。
- 超参预算（两模型相同，只用 validation Rank IC 选择）：
  - Phase 1：patience=1 作为快速筛选，依次扫 lr ∈ {3e-6, 1e-5, 3e-5, 1e-4} 与 dropout ∈ {0.4, 0.5, 0.6}，再在选中组合上跑 patience ∈ {2, 5}，并加入 Jiang 锚点 (1e-5, 0.5, 2)。
  - Phase 2：只在预注册 patience ∈ {2, 5} 的候选里选 validation Rank IC 最高者。
  - 最终：选中配置下 seeds 42–46 五模型 ensemble，做 18 年 OOS 预测。
- 评估：与完整 chart-CNN、1D-CNN-120、MLP-4-120 在共同 OOS 样本上比较（2,687,822 个股票-周，922 周）。指标为 AUC、IC、Rank IC（NW t，lag=4）、十分位 EW/VW H-L 年化收益与 Sharpe（ann=52）。

## 3. 首轮运行作废说明

首轮（2026-09-23）结果无效，已移至 `WORK_SPACE/paper/E2G_chart_capacity_invalid_fullwidth/`：

- `CNNModel.__init__` 按 `window_size` 分支时无条件覆盖了传入的 `channels`，两个“缩放”模型实际都是 64-128-256（708,866 参数）。
- Phase 2 允许 patience=1 的筛选行进入最终候选，M 最终选中了预注册网格之外的 patience=1。

修复内容：
- `CNNModel` 仅在 `channels is None` 时使用默认宽度。其他实验都不传 `channels`，不受影响。
- `run_e2g.py` 训练前断言实际宽度与参数量等于 `expected_parameters`。
- Phase 2 只保留 patience ∈ {2, 5} 的候选。

重跑后 checkpoint 为 S 约 40 KB、M 约 109 KB（首轮均为 2.8 MB）。

## 4. 选中配置与训练诊断

| 模型 | Channels | 参数（conv / BN / head / 总） | lr | dropout | patience | 选中 val Rank IC | 5 seed 平均 val Rank IC | train–val loss gap | best / stop epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| ChartCNN-S | 4-8-16 | 2,488 / 56 / 5,762 / 8,306 | 1e-4 | 0.6 | 5 | 0.0537 | 0.0519 | 0.021 | 3.0 / 9.6 |
| ChartCNN-M | 9-18-36 | 12,348 / 126 / 12,962 / 25,436 | 3e-5 | 0.6 | 5 | 0.0517 | 0.0479 | 0.017 | 5.8 / 10.0 |

缩放后 conv 与 head 两部分同时缩小，参数量变化不只是分类头的机械变化。两个模型的 train–val gap 都很小，也都早停于前 10 个 epoch，没有明显过拟合迹象。

## 5. OOS 结果（共同样本）

| 模型 | 参数 | AUC | Rank IC | NW t | EW H-L 年化 | EW Sharpe | VW H-L 年化 | VW Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ChartCNN-S | 8,306 | 0.5206 | 0.0307 | 10.27 | 27.9% | 2.32 | 20.0% | 1.10 |
| ChartCNN-M | 25,436 | 0.5201 | 0.0288 | 9.78 | 24.5% | 2.16 | 17.2% | 0.97 |
| Chart-CNN（完整） | 708,866 | 0.5190 | 0.0308 | 11.72 | 23.6% | 2.39 | 18.0% | 1.11 |
| 1D-CNN-120 | 7,073 | 0.5261 | 0.0484 | 13.42 | 52.2% | 3.21 | 37.6% | 1.64 |
| MLP-4-120 | 26,369 | 0.5196 | 0.0410 | 14.43 | 36.5% | 2.81 | 22.9% | 1.32 |

> 更正（E0 对账，2026-10）：此前 MLP-4-120 一行误用了 180D bag-10 的预测文件（`E2E_mlp_bagging/preds/pred_mlp4_180_bag10_oos.csv`）。现已改为规范的 120D seed-5 文件 `E2F_mlp_depth/preds/pred_mlp4_oos.csv`（见 `docs/protocol/model_registry.md`），结论不变。

与基准的平均截面 Spearman 相关（逐周计算后取平均）：

| 模型 | 与完整 Chart-CNN | 与 1D-CNN-120 | 与 MLP-4-120 |
|---|---:|---:|---:|
| ChartCNN-S | 0.80 | 0.38 | 0.36 |
| ChartCNN-M | 0.85 | 0.37 | 0.36 |

## 6. 结论分析

1. **容量失配不能解释 chart-CNN 的劣势。** 参数量缩小约 85 倍（S）或 28 倍（M）之后，chart-CNN 的 OOS 表现基本不变：
   - S 与完整模型几乎相同（Rank IC 0.0307 vs 0.0308，EW Sharpe 2.32 vs 2.39，VW Sharpe 1.10 vs 1.11）；
   - M 略低（Rank IC 0.0288，EW Sharpe 2.16）。
   原模型的大参数量既没有带来收益，也不是性能差距的来源。
2. **同参数量下差距依旧。**
   - ChartCNN-S 与 1D-CNN-120 参数量接近（8,306 vs 7,073）：Rank IC 0.031 vs 0.048，EW Sharpe 2.32 vs 3.21。
   - ChartCNN-M 与 MLP-4-120 参数量接近（25,436 vs 26,369）：Rank IC 0.029 vs 0.041，EW Sharpe 2.16 vs 2.81。
   差距的量级与完整模型对比时相同。
3. **缩放后学到的仍是同一个信号。** S/M 与完整 chart-CNN 的相关为 0.80–0.85，与 1D-CNN/MLP-4 的相关仅约 0.36–0.38，与完整模型和数值模型之间的关系相同。宽度变化没有改变 chart-CNN 提取的信号类型。
4. **validation 与 OOS 的排序不一致。** S/M 的 validation Rank IC（0.048–0.054）与完整模型接近，但 OOS 都在 0.03 左右。chart-CNN 在 2006–2007 validation 上的优势外推到 2008–2025 时衰减较多。

## 7. 判读边界

E2G 只控制了可训练参数量，没有匹配 FLOPs、activation size、优化难度或函数类：

> The exercise isolates one important dimension of capacity mismatch rather than claiming complete architectural equivalence.

E2G 只比较了单模型表现。ChartCNN-S 在传统控制变量和 1D-CNN 下的增量检验见 [paper_results_E7.md](paper_results_E7.md) 第 10 节：S 与完整模型一样被 1D-CNN 包含。

结果支持的表述是：参数量失配不是 chart-CNN 弱于数值模型的合理解释。剩下的差距更可能来自表示方式（图像 vs 数值序列）、二维归纳偏置或优化难度。E2G 不能区分这几种来源。设计文档中的对称检验（按同一规则扩大 1D-CNN）尚未运行。
