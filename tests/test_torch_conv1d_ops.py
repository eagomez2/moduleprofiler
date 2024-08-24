import pytest
import torch
import torch.nn as nn


@pytest.mark.parametrize(
        "batch_size, in_features, in_channels, out_channels, kernel_size, "
        "groups, bias", [
            # batch_size=1
            (1, 8, 1, 1, 3, 1, False),
            (1, 9, 1, 1, 3, 1, True),

            # batch_size=8
            (8, 8, 2, 4, 2, 1, False),
            (8, 9, 2, 4, 2, 1, True),

            # batch_size=8 and groups=2
            (8, 8, 2, 4, 2, 2, False),
            (8, 9, 2, 4, 2, 2, True)
        ]
)
def test_conv1d_simplified_output_formula(
    batch_size: int,
    in_features: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    groups: int,
    bias: bool
) -> None:
    # Sample input tensor
    x = torch.rand((batch_size, in_channels, in_features))
    
    # Built-in module
    net = nn.Conv1d(
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
    filter_ops = y.size(-1) * (2 * kernel_size - 1)
    filter_aggr_ops = (
        out_channels * y.size(-1) * (in_channels / groups) if bias
        else out_channels * y.size(-1) * ((in_channels / groups) - 1) 
    )
    total_ops = int(batch_size * (num_filters * filter_ops + filter_aggr_ops))

    # Simplified formula
    if bias:
        simplified_ops_num = (
            out_channels * y.size(-1)
            * in_channels * 2 * kernel_size
        )

    else:
        simplified_ops_num = (
            out_channels * y.size(-1)
            * (in_channels * 2 * kernel_size - groups)
        )
    
    simplified_total_ops = int(batch_size * (simplified_ops_num / groups))

    assert total_ops == simplified_total_ops 
