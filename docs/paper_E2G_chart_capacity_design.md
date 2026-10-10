# E2G：参数量匹配的 Chart-CNN 容量控制

## 研究问题

原 Jiang-style 20 日 chart-CNN 在当前 A 股样本中相对 1D-CNN 与 MLP-4 的较弱表现，是否主要来自 trainable-parameter scale 的失配，而非图像表示或二维归纳偏置本身？

E2G 是参数量控制实验，不是寻找最强新图像模型。

## 固定不变的部分

两个 E2G 模型均保留原 I20 Jiang-style chart-CNN 的：图像 renderer、`1 x 64 x 60` 输入、20 日窗口、未来五日二分类标签、三层卷积深度、`5 x 3` kernel、第一层 stride/dilation、BatchNorm、LeakyReLU、`2 x 1` max pooling、flatten head、时间切分、validation Rank IC 选择与 OOS 评价。

唯一的架构变化是各卷积 block 的 channel width。

## 预先定义的宽度规则

不采用手工的 `[8,16,40]`。只允许严格比例缩放：

`[64,128,256] -> [k,2k,4k]`。

对每一目标参数量，取 `k in {1,...,64}` 中使 `|P(k)-P_target|` 最小者；不参考 validation 或 OOS 结果。对固定 I20 `64 x 60`、两分类架构，

`P(k) = 150 k^2 + 1476 k + 2`。

| 模型 | 目标基准 | 目标参数量 | k | Channels | 实际参数量 | 差异 |
|---|---|---:|---:|---|---:|---:|
| ChartCNN-S | 1D-CNN-120 | 7,073 | 4 | 4-8-16 | 8,306 | +1,233 |
| ChartCNN-M | MLP-4-120 | 26,369 | 9 | 9-18-36 | 25,436 | -933 |

参数分解必须同时报告为 `P_conv`、`P_BN`、`P_head`，以说明缩放不只是最后分类头的机械变化。

## 训练公平性

- 两个宽度缩放版本使用完全相同的、validation-only 的 chart-CNN 超参数预算：learning rate `{3e-6, 1e-5, 3e-5, 1e-4}`、dropout `{0.4,0.5,0.6}`、patience `{2,5}`。
- `2000--2005` 训练，`2006--2007` validation，`2008--2025` OOS；不得用 OOS 选择宽度或训练参数。
- 每种结构先完成同预算的单 seed validation search；固定选择后均使用 seeds `42--46` 的五模型 ensemble。
- lr 与 dropout 的初筛用 patience=1 加速；最终候选只允许预注册的 patience `{2,5}`。
- 训练前断言实际 channels 与参数量等于上表；首轮因 `CNNModel` 覆盖 `channels` 而作废（见 `docs/paper_results_E2G.md`）。
- 每个 seed 保存每 epoch train loss、validation loss、validation Rank IC、best/stop epoch，以报告 train--validation gap。

## 报告与判读边界

主表报告 parameter count、分解、validation Rank IC、OOS Rank IC、EW/VW H-L Sharpe、训练--验证 gap 与 early-stop epoch。使用五 seed OOS 平均分数。

E2G 只控制 trainable parameter count，不等同于匹配 FLOPs、activation size、优化难度或函数类。因此可表述为：

> The exercise isolates one important dimension of capacity mismatch rather than claiming complete architectural equivalence.

不预先限定结果解释的文字结论。结果应结合 validation/OOS Rank IC、EW/VW Sharpe、训练--验证 gap、early-stop epoch 和参数分解共同判读。参数量控制可以检验容量失配是否是一个合理解释，但不能单独证明图像表示、函数类或优化难度是任一性能差异的唯一原因。

## 后续对称检验

首轮先运行 ChartCNN-S 与 ChartCNN-M。若需要更强的 capacity--representation 识别，再预先设计 1D-CNN-L，使其按相同规则扩大，但不在首轮结果后临时选宽度。
