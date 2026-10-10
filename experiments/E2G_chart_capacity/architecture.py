"""Pre-specified proportional-width rule for E2G chart-CNN capacity controls."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChartCapacitySpec:
    name: str
    target_parameters: int
    width_multiplier: int
    channels: tuple[int, int, int]
    parameter_error: int


def parameter_breakdown(k: int, image_height: int = 64, image_width: int = 60, n_classes: int = 2) -> dict[str, int]:
    """Exact parameters for the fixed I20 Jiang-style CNN under [k, 2k, 4k]."""
    if image_height != 64 or image_width != 60 or n_classes != 2:
        raise ValueError('E2G rule is pre-specified for the fixed I20 64x60, two-class architecture.')
    c1, c2, c3 = k, 2 * k, 4 * k
    # After the fixed stride/dilation and three 2x1 pools, I20 feature map is c3 x 3 x 60.
    conv = (1 * c1 * 5 * 3 + c1) + (c1 * c2 * 5 * 3 + c2) + (c2 * c3 * 5 * 3 + c3)
    batch_norm = 2 * (c1 + c2 + c3)
    head = (c3 * 3 * 60) * n_classes + n_classes
    return {'conv': conv, 'batch_norm': batch_norm, 'head': head, 'total': conv + batch_norm + head}


def nearest_proportional_spec(name: str, target_parameters: int, k_min: int = 1, k_max: int = 64) -> ChartCapacitySpec:
    """Select k mechanically: minimize |P(k)-target| over integer proportional widths."""
    candidates = []
    for k in range(k_min, k_max + 1):
        total = parameter_breakdown(k)['total']
        candidates.append((abs(total - target_parameters), k, total))
    error, k, total = min(candidates, key=lambda x: (x[0], x[1]))
    return ChartCapacitySpec(
        name=name,
        target_parameters=target_parameters,
        width_multiplier=k,
        channels=(k, 2 * k, 4 * k),
        parameter_error=total - target_parameters,
    )


ONE_D_CNN_PARAMETERS = 7_073
MLP4_PARAMETERS = 26_369
CHART_SMALL = nearest_proportional_spec('ChartCNN-S', ONE_D_CNN_PARAMETERS)
CHART_MEDIUM = nearest_proportional_spec('ChartCNN-M', MLP4_PARAMETERS)


if __name__ == '__main__':
    for spec in (CHART_SMALL, CHART_MEDIUM):
        parts = parameter_breakdown(spec.width_multiplier)
        print(spec)
        print(parts)
