# WORK_SPACE（物理存储在 F 盘）

本目录为 **目录联接（junction）**，物理路径：

```
F:\LLM_trading\CNN_trading\WORK_SPACE\
```

项目代码中统一使用相对路径 `WORK_SPACE/...`（或 `Misc/config.py` 中的 `WORK_DIR`），无需改各实验 yaml。

## 目录说明

| 子目录 | 内容 |
|--------|------|
| `data/` | 预处理 feather、图表数据集（~22GB） |
| `ashare_model_res_I20R5/` 等 | CNN 训练产出（ensem、portfolio） |
| `paper/` | 论文实验（E1b、E2 等） |

## 环境变量（可选）

若需指向其他盘符，可设置：

```powershell
$env:CNN_TRADING_WORK_DIR = "D:\your_path\WORK_SPACE"
```

默认不设置时，`WORK_DIR = PROJECT_ROOT/WORK_SPACE`（junction → F 盘）。
