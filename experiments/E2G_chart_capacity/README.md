# E2G: Parameter-Count-Matched Chart-CNN Control

E2G tests whether the large Jiang-style I20 chart CNN's relative weakness can be
explained by trainable-parameter scale. It is not a search for a new chart architecture.

The renderer, input image, three convolutional blocks, kernels, stride/dilation,
BatchNorm, LeakyReLU, pooling, flatten head, label, chronological split, and OOS
evaluation remain fixed. Only widths vary under the pre-specified proportional rule
`[k, 2k, 4k]`.

`architecture.py` mechanically selects integer `k` by minimizing the absolute difference
between trainable parameter count and the target benchmark. This yields:

- `ChartCNN-S = [4, 8, 16]`, 8,306 parameters, targeted at 1D-CNN's 7,073;
- `ChartCNN-M = [9, 18, 36]`, 25,436 parameters, targeted at MLP-4's 26,369.

Both variants receive the same validation-only chart-CNN hyperparameter budget and
the same five final random seeds. E2G matches trainable parameter count only; it does
not claim to match FLOPs, activation size, optimization difficulty, or function class.
