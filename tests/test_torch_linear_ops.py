import pytest
import torch
import torch.nn as nn


@pytest.mark.parametrize(
        "in_size, out_features, bias", [
            # 1d case
            ((1,), 1, False),
            ((1,), 1, True),

            # 2d case
            ((2, 2), 2, False),
            ((2, 2), 2, True),

            # 3d case
            ((3, 3), 3, False),
            ((3, 3), 3, True),

            # Non-symmetric additional cases
            ((2, 3, 4), 8, False),
            ((2, 3, 4), 8, True),
        ]
)
def test_linear_output_match(
    in_size: tuple,
    out_features: int,
    bias: bool
) -> None:
    # Sample tensor
    x = torch.rand(in_size)

    # Built-in module
    net = nn.Linear(
        in_features=in_size[-1],
        out_features=out_features,
        bias=bias
    )
    
    # Automatic calculation
    y_torch = net(x)

    # Manual calculation
    y_mp = x @ net.weight.T
    
    if bias:
        y_mp += net.bias
    
    # NOTE: Computing all with pytorch function may differ from the underlying
    # optimized module implementation so small differences are expected
    assert torch.allclose(y_torch, y_mp)
