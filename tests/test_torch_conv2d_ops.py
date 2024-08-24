import pytest
import torch
import torch.nn as nn
from typing import Tuple


@pytest.mark.parametrize(
        "batch_size, in_h_features, in_w_features, in_channels, out_channels, "
        "kernel_size, groups, bias", [
            # batch_size=1
            (1, 8, 8, 1, 1, (1, 1), 1, False),
            (1, 8, 8, 1, 1, (1, 2), 1, False),
            (1, 8, 8, 1, 1, (2, 1), 1, False),
            (1, 8, 8, 1, 1, (2, 2), 1, False),
            (1, 8, 8, 1, 1, (2, 2), 1, True),
            (1, 8, 8, 1, 1, (1, 2), 1, True),
            (1, 8, 8, 1, 1, (2, 1), 1, True),
            (1, 8, 8, 1, 1, (2, 2), 1, True),

            # batch_size=8
            (8, 8, 8, 1, 1, (1, 1), 1, False),
            (8, 8, 8, 1, 1, (1, 2), 1, False),
            (8, 8, 8, 1, 1, (2, 1), 1, False),
            (8, 8, 8, 1, 1, (2, 2), 1, False),
            (8, 8, 8, 1, 1, (1, 1), 1, True),
            (8, 8, 8, 1, 1, (1, 2), 1, True),
            (8, 8, 8, 1, 1, (2, 1), 1, True),
            (8, 8, 8, 1, 1, (2, 2), 1, True),

            # batch_size=8 and groups=2
            (8, 8, 8, 2, 4, (1, 1), 2, False),
            (8, 8, 8, 2, 4, (1, 2), 2, False),
            (8, 8, 8, 2, 4, (2, 1), 2, False),
            (8, 8, 8, 2, 4, (2, 2), 2, False),
            (8, 8, 8, 2, 4, (1, 1), 2, True),
            (8, 8, 8, 2, 4, (1, 2), 2, True),
            (8, 8, 8, 2, 4, (2, 1), 2, True),
            (8, 8, 8, 2, 4, (2, 2), 2, True),
        ]
)
def test_conv2d_simplified_output_formula(
        batch_size: int,
        in_h_features: int,
        in_w_features: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int],
        groups: int,
        bias: bool
) -> None:
    # Sample input tensor
    x = torch.rand((batch_size, in_channels, in_h_features, in_w_features))

    # Built-in module
    net = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        bias=bias
    )

    # Automatic calculation
    y = net(x)

    # Step by step formula
    num_filters = (in_channels * out_channels) / groups
    filter_ops = (
        y.size(-1) * y.size(-2) * (2 * kernel_size[0] * kernel_size[1] - 1)
    )

    if bias:
        filter_aggr_ops = (
            out_channels * y.size(-1) * y.size(-2) * (in_channels / groups)
        )
    
    else:
        filter_aggr_ops = (
            out_channels * y.size(-1) * y.size(-2)
            * ((in_channels / groups) - 1)
        )

    total_ops = int(batch_size * (num_filters * filter_ops + filter_aggr_ops))

    # Simplified formula
    if bias:
        simplified_ops_num = (
            out_channels * in_channels * y.size(-1) * y.size(-2)
            * (2 * kernel_size[0] * kernel_size[1])
        )
    
    else:
        simplified_ops_num = (
            out_channels * y.size(-1) * y.size(-2)
            * (in_channels * 2 * kernel_size[0] * kernel_size[1] - groups)
        )

    simplified_total_ops = int(batch_size * (simplified_ops_num / groups))

    assert total_ops == simplified_total_ops
