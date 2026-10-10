# -*- coding: utf-8 -*-
"""验证 LegacyCNNModel 架构与 checkpoint 加载"""
import sys
import torch
import importlib.util

sys.path.insert(0, 'trend_submit/trend_code_submit')
sys.path.insert(0, '.')

spec = importlib.util.spec_from_file_location('pred_mod', '5_predict_with_model_res_0.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

LegacyCNNModel = mod.LegacyCNNModel
load_ensemble = mod.load_ensemble

device = torch.device('cpu')
print("=== 验证 LegacyCNNModel 架构 ===")
m = LegacyCNNModel(dropout=0.0)
fc_in = m.fc.in_features
ok = "OK" if fc_in == 46080 else "FAIL"
print(f"  fc_in = {fc_in}  (期望 46080) -> {ok}")
print(f"  总参数量: {sum(p.numel() for p in m.parameters()):,}")

print()
print("=== 加载 model_res_0 checkpoints ===")
models = load_ensemble('WORK_SPACE/model_res_0', 5, device)
print(f"  成功加载: {len(models)} 个模型")

print()
print("=== 前向测试（随机图像 batch=4, 1x64x60）===")
dummy = torch.randn(4, 1, 64, 60)
models[0].eval()
with torch.no_grad():
    out = models[0].predict_proba(dummy)
print(f"  输出形状: {out.shape}  (期望 torch.Size([4, 2]))")
print(f"  概率示例: {out[0].tolist()}")
print(f"  概率和: {float(out[0].sum()):.4f}  (期望 1.0)")

shape_ok = out.shape == torch.Size([4, 2])
prob_ok = abs(float(out[0].sum()) - 1.0) < 1e-5
if shape_ok and prob_ok and fc_in == 46080 and len(models) == 5:
    print("\n所有验证通过！可以运行: python 5_predict_with_model_res_0.py")
else:
    print("\n验证失败，请检查上方错误信息！")





