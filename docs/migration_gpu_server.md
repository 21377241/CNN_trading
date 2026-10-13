# 迁移到 GPU 服务器（H100）训练

只迁移训练所需的代码和数据。E7、E9、E10 等分析只用 CPU，可以把新机器上生成的预测文件拉回本机再跑。

## 1. 代码与环境

```bash
git clone -b clean-data https://github.com/21377241/CNN_trading.git
cd CNN_trading

conda create -n cnn python=3.12.3 -y && conda activate cnn
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

`scikit-learn` 的版本须与本机相同（1.7.2），否则 joblib 保存的 MLP 可能无法读取或预测不同。

## 2. 数据

所有数据路径都由 `WORK_SPACE` 推出（`trend_submit/trend_code_submit/Misc/config.py`）：默认是仓库根目录下的 `WORK_SPACE/`，也可以用环境变量 `CNN_TRADING_WORK_DIR` 指向别处（绝对路径）。

### 2.1 必须迁移（约 17.5 GB）

| 内容 | 在 WORK_SPACE 下的路径 | 文件数 | 大小 |
|---|---|---|---|
| chart-CNN 图像与标签（2000–2026，每年 `_images.dat` 与 `_labels.feather`） | `data/ashare_stocks_dataset/I20R5_clean/` | 54 | 13.4 GB |
| 140 维特征（`X_is.npy`、`X_oos.npy`、`meta_*.feather`、`by_year/`） | `paper/clean_inputs/features140/` | 32 | 4.0 GB |
| universe（样本键与 train/val/oos 切分） | `data/ashare_processed_data/universe_weekly.feather` | 1 | 0.1 GB |
| 文件清单与 sha1 | `migration_manifest.json` | 1 | 很小 |

目录结构须保持不变。本机打包（在仓库根目录，PowerShell 或 bash 均可）：

```bash
tar -cf cnn_data.tar -C WORK_SPACE \
  data/ashare_stocks_dataset/I20R5_clean \
  paper/clean_inputs/features140 \
  data/ashare_processed_data/universe_weekly.feather \
  migration_manifest.json
```

新机器上解包并逐个文件核对：

```bash
mkdir -p WORK_SPACE && tar -xf cnn_data.tar -C WORK_SPACE
python ashare_train/data_manifest.py --check     # 须显示 87/87 个文件一致
```

### 2.2 按需迁移（很小）

做扩展窗口或稳健性检验时，超参数沿用已选定的组合，需要这些记录：

| 内容 | 在 WORK_SPACE 下的路径 |
|---|---|
| registry v2 | `paper/registry_v2.json` |
| 数值模型选定记录 | `paper/hp_numeric_140/{mlp,oned}/selection.json`、`phase1.csv`、`phase2.csv` |
| chart-CNN 搜参记录 | `paper/E1c_cnn_hp_I20R5_clean/search_phase1.csv`、`search_phase2.csv`、`phase3_full_train.csv`、`config_cnn_hp.yaml` |

### 2.3 不需要迁移

- `WORK_SPACE/_olddata/`（46 GB）：只用于 M0、M1 的对照，已经跑完。
- CSMAR 面板与其余 `data/ashare_processed_data/` 文件：只在重新生成图像、特征或 E7 控制变量时才用。
- 已训练的模型与已登记的预测（`paper/registry_v2/`）。

## 3. 注意事项

- 不同 GPU 上训练的结果不会逐位相同。已登记的三个模型保持本机结果，不在新机器上重训替换。新机器上做的实验（扩展窗口、稳健性检验）全部在新机器上完成，并在 `docs/protocol/clean_spec.md` 中记录 GPU 型号与软件版本。
- 精度设置保持 PyTorch 默认（卷积默认用 TF32，本机 RTX 3060 同样如此；矩阵乘法默认不用 TF32），不加混合精度；batch 仍为 128。与本机的设定一致。
- `tests/test_clean_independence.py` 会检查工作区内容；新机器上只有部分数据，与工作区相关的检查可能不通过，这是预期的。
