# 新环境（H100 容器）交接说明

写于 2026-10-13。给在新容器上继续工作的人（或 Cursor agent）。规则的唯一依据是 `docs/protocol/clean_spec.md`；本文件只说明现状、必须遵守的约束和执行顺序。

## 1. 现状

- 分支 `clean-data`。数据源为 CSMAR；图像渲染与 Jiang、Kelly & Xiu（JKX）官方代码逐像素一致。
- **S0 切分已完成并登记**（训练 2000–2005、验证 2006–2007、OOS 2008–2026.9）：`chartcnn_clean`、`onedcnn_140`、`mlp_140`，见 `WORK_SPACE/paper/registry_v2.json` 和 clean_spec 第 9 节“2026-10-13 C3 完成”。
- **主分析已改为扩展窗口（EW）**，S0 与 S1 为稳健性检验，E2G、E2F 也在 EW 上做：clean_spec 第 9 节“2026-10-13 主分析改为扩展窗口”。EW 与 S1 都还没有任何训练。
- 干净数据上**还没有计算过任何 OOS 指标**。
- C3 影响报告（M1 − M0、M2 − M1）在原机器上做，新环境不做。两台机器的分工见第 6 节。

## 2. 必须遵守的约束

1. **读规格书的方式**：第 9 节按日期追加，后面的记录覆盖前面；正文（第 1–8 节）中被修订的条目有“（已修订：……）”注释。当前有效的关键规则：
   - 所有模型按验证 Rank IC 早停（`monitor: rank_ic`）：第 9 节“2026-10-12 早停规则修正”。
   - chart-CNN 搜参：第 5.1 节网格与三阶段流程，`experiments/E1_image_baseline/config_cnn_hp_clean.yaml`；第三阶段选定规则见“2026-10-12 C3 训练前补充”。
   - 数值模型搜参：“2026-10-12 数值模型的搜参”（网格、两阶段、选定与并列规则）。
   - EW、S1、E2G、E2F：“2026-10-13 主分析改为扩展窗口”。
2. **OOS**：在 C4 打 tag `protocol-clean-v1` 之前，不得在 OOS 年份上计算任何指标（Rank IC、Sharpe、收益、回归等）。训练只输出验证集指标和 OOS 预测；登记时只做核对（键、取值范围、由权重重新预测的差）。
3. **先写规则，再运行**：规格书没有写到的细节（目录命名、并行方式、每次重训的具体流程等），先在 clean_spec 第 9 节新增一段带日期、注明“写于……之前”的记录，提交并推送，然后才运行。
4. **遇到以下情况先停下来问用户**：规格书没有覆盖的决定；结果与预期明显不符（如环境等价性检查不通过、搜参中某组全部停在第 1 轮）；需要改动已预注册的规则。改动规则时，在第 9 节如实记录起因、决定时已看到的结果，作废的结果移入 `WORK_SPACE/paper/_sup<日期>/`，不删除。
5. **不动 S0**：不覆盖、不重训已登记的 S0 模型与预测。新结果写到新目录，目录名须包含切分标记（如 `ew2011`、`s1`），不得与 S0 共用缓存。读取已有 checkpoint 时代码会校验 `monitor`。
6. **可复现**：seed 42–46；batch 与精度设置不变（PyTorch 默认，卷积用 TF32，不加混合精度）；硬件、驱动、软件版本记入 clean_spec。
7. **沟通与提交**：用简体中文汇报；提交说明用 UTF-8（无 BOM）；每完成一步提交并推送。

## 3. 环境准备

按 `docs/migration_gpu_server.md`：

1. GPU：`nvidia-smi` 须能显示 H100（2026-10-13 时容器报 “Failed to initialize NVML”，需平台修复）。
2. 数据包 `/root/cnn_trading/cnn_data.tar.gz` 的 sha1 须为 `d3dd0ad56c11c14a08ef7858ed263a1a0a1155fb`（2026-10-13 时只传了约 96%，需续传）。
3. 代码：`/root/cnn_trading` 已有数据包，非空目录不能直接 clone，用 `git init`、`git remote add origin <url>`、`git fetch origin clean-data`、`git checkout -b clean-data origin/clean-data`。
4. 安装环境、解包到 `WORK_SPACE/`，`python ashare_train/data_manifest.py --check` 须为 98/98。

## 4. 执行顺序

**A. 环境等价性检查**

- 在 H100 上重训 S0 三个选定组合各 seed 42，输出到 `WORK_SPACE/paper/env_check/`（不得写入 S0 目录）。
- 与 S0 的 seed 42 比较验证 Rank IC（chart-CNN 0.0472、1D-CNN 0.0751、MLP 0.0598）。不同 GPU 不会逐位相同；差值的判定标准先写进 clean_spec 再运行（参考：S0 五个 seed 之间的差异，chart-CNN 0.038–0.053、1D-CNN 0.072–0.078、MLP 0.052–0.065）。
- 同时记录每轮耗时，用于估计后续时间。

**B. 代码（每项先用小样本试跑，再提交）**

1. 切分可配置：chart-CNN 搜参与完整训练（`ashare_train/3b_cnn_hp_search.py`、`3c_cnn_full_train.py`、配置中的年份与 `out_root`），数值模型搜参（`ashare_train/3d_numeric_hp_search.py` 的 `TRAIN_YEARS`、`VAL_YEARS`、`OUT_BASE` 目前是常量）。embargo 须按每次切分的边界重新确定（universe 中现有的 `embargo` 列只对应 S0 的边界）。
2. EW 驱动：按重训年 2011、2014、2017、2020、2023 依次搜参、训练、预测；与 S0（重训年 2008）按信号年份拼接；登记为 `chartcnn_ew`、`onedcnn_140_ew`、`mlp_140_ew`，记录每段来源。
3. S1：训练 2000–2011、验证 2012–2013，登记为 `*_s1`。
4. 并行调度：同一块 GPU 上同时跑多个试验或 seed，只改调度，不改训练逻辑。
5. E2G 改写为干净数据流程（`experiments/E2G_chart_capacity/run_e2g.py` 仍是旧流程）：每次重训的 ChartCNN-S/M、1D-CNN-L；E2F（`experiments/E2F_mlp_depth/run_depth.py` 或在 `3d_numeric_hp_search.py` 基础上实现）。
6. 登记：扩展 `ashare_train/build_registry_v2.py`，每个新模型由权重重新预测核对。

**C. 训练**：EW（主分析）、S1、E2G、E2F。只输出验证集指标与 OOS 预测，完成后登记。

**D. C4（须用户审阅后打 tag）**：更新 E7 v2、E9、E10 协议（主分析改用 EW 的预测，模型名称，`experiments/E7_incremental/common.py` 中 `s_mlp4` 键改名），写稳健性检验协议（北交所、n_obs ≥ 15、满 20 天、S0、S1、E2G、E2F 及“结论稳健”的判定标准），打 tag `protocol-clean-v1`。

**E. C5**：打 tag 之后才开始。需要第二个数据包（CSMAR 面板、E7 输入、登记的预测），由原机器提供。

## 5. 关键文件

| 用途 | 文件 |
|---|---|
| 规则 | `docs/protocol/clean_spec.md` |
| chart-CNN 训练 | `trend_submit/trend_code_submit/Experiments/cnn_experiment.py`、`ashare_train/3b_cnn_hp_search.py`、`ashare_train/3c_cnn_full_train.py` |
| 数值模型 | `ashare_train/3d_numeric_hp_search.py`、`experiments/E2C_temporal/train_1dcnn.py`、`experiments/E2_matched_tabular/mlp_bagging.py` |
| 登记 | `ashare_train/build_registry_v2.py` |
| 数据清单 | `ashare_train/data_manifest.py` |
| 路径配置 | `trend_submit/trend_code_submit/Misc/config.py`（`WORK_SPACE` 或环境变量 `CNN_TRADING_WORK_DIR`） |

## 6. 两台机器的分工与同步

代码同时在原机器（Windows，RTX 3060）和容器上修改，通过 GitHub 的 `clean-data` 分支同步。

| | 负责 | 修改的文件 |
|---|---|---|
| 容器 | 第 4 节 A–E 的全部代码与训练 | `ashare_train/`、`experiments/`、`trend_submit/` 下的代码与配置（下面列给原机器的除外），新增的协议文档 |
| 原机器 | C3 影响报告；第二个数据包及其清单 | `experiments/E1_image_baseline/eval_clean_cnn_impact.py`、`docs/figures/clean_cnn_impact_*`、`docs/results/clean_cnn_impact.md`、`ashare_train/data_manifest.py` |

- 两边都会在 clean_spec 第 9 节末尾追加记录。
- 每次开始工作前先 `git pull --rebase origin clean-data`。小步提交，提交后立即推送；不用 `--force`。
- clean_spec 冲突时保留双方的记录，按日期排列，不改对方的内容。
- 需要改对方负责的文件时，先告诉用户。
- `WORK_SPACE/` 不进 git。训练产物留在容器上；需要给原机器的结果，按 `data_manifest.py` 的方式打包并核对 sha1。
