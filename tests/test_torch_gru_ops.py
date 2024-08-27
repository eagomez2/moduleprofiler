import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


@pytest.mark.parametrize(
        "input_size, hidden_size", [
            (1, 1),
            (8, 16),
            (16, 8)
        ]
)
def test_gru_output_match_no_bias_one_layer_two_steps_unidirectional(
        input_size: int,
        hidden_size: int
) -> None:
    # Sample input and hidden state tensors
    x = torch.rand((2, input_size), dtype=torch.float32)
    hx = torch.rand((1, hidden_size), dtype=torch.float32)

    # Built-in module
    net = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=False,
        dropout=0.0,
        bidirectional=False
    )

    # Automatic calculation
    y_torch, h_torch = net(x, hx)

    # Manual calculation (weight order ir, iz, in)
    # (hidden_size, input_size)
    weight_ih_r_l0 = net.weight_ih_l0[0:hidden_size, ...]
    weight_ih_z_l0 = net.weight_ih_l0[hidden_size:(2 * hidden_size), ...]
    weight_ih_n_l0 = net.weight_ih_l0[(2 * hidden_size):(3 * hidden_size), ...]
    weight_hh_r_l0 = net.weight_hh_l0[0:hidden_size, ...]
    weight_hh_z_l0 = net.weight_hh_l0[hidden_size:(2 * hidden_size), ...]
    weight_hh_n_l0 = net.weight_hh_l0[(2 * hidden_size):(3 * hidden_size), ...]

    # First step
    x_s0 = torch.narrow(x, dim=0, start=0, length=1)

    r_s0_l0 = F.sigmoid(weight_ih_r_l0 @ x_s0.T + weight_hh_r_l0 @ hx.T)
    z_s0_l0 = F.sigmoid(weight_ih_z_l0 @ x_s0.T + weight_hh_z_l0 @ hx.T)
    n_s0_l0 = F.tanh(
        weight_ih_n_l0 @ x_s0.T + r_s0_l0 * (weight_hh_n_l0 @ hx.T)
    )
    h_s1 = ((1.0 - z_s0_l0) * n_s0_l0 + z_s0_l0 * hx.T).T

    # Second step
    x_s1 = torch.narrow(x, dim=0, start=1, length=1)

    r_s1_l0 = F.sigmoid(weight_ih_r_l0 @ x_s1.T + weight_hh_r_l0 @ h_s1.T)
    z_s1_l0 = F.sigmoid(weight_ih_z_l0 @ x_s1.T + weight_hh_z_l0 @ h_s1.T)
    n_s1_l0 = F.tanh(
        weight_ih_n_l0 @ x_s1.T + r_s1_l0 * (weight_hh_n_l0 @ h_s1.T)
    )

    h_s2 = ((1.0 - z_s1_l0) * n_s1_l0 + z_s1_l0 * h_s1.T).T

    # Get final output
    y_mp = torch.cat((h_s1, h_s2), dim=0)

    assert (
        torch.allclose(y_torch, y_mp, atol=1e-5)
        and torch.allclose(h_torch, h_s2, atol=1e-5)
    )


@pytest.mark.parametrize(
        "input_size, hidden_size", [
            (1, 1),
            (8, 16),
            (16, 8)
        ]
)
def test_gru_output_match_no_bias_one_layer_two_steps_bidirectional(
        input_size: int,
        hidden_size: int
) -> None:
    # Sample input and hidden state tensors
    x = torch.rand((2, input_size), dtype=torch.float32)
    hx = torch.rand((2, hidden_size), dtype=torch.float32)

    # Built-in module
    net = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=False,
        dropout=0.0,
        bidirectional=True
    )

    # Automatic calculation
    y_torch, h_torch = net(x, hx)

    # Manual calculation (weight order ir, iz, in)
    # (hidden_size, d * input_size)
    weight_ih_r_l0f = net.weight_ih_l0[0:hidden_size, 0:1]
    weight_ih_z_l0f = net.weight_ih_l0[hidden_size:(2 * hidden_size), 0:1]
    weight_ih_n_l0f = net.weight_ih_l0[
        (2 * hidden_size):(3 * hidden_size), 0:1
    ]
    weight_hh_r_l0f = net.weight_hh_l0[0:hidden_size, 0:1]
    weight_hh_z_l0f = net.weight_hh_l0[hidden_size:(2 * hidden_size), 0:1]
    weight_hh_n_l0f = net.weight_hh_l0[
        (2 * hidden_size):(3 * hidden_size), 0:1
    ]
    
    # First step forward
    x_s0f = torch.narrow(x, dim=0, start=0, length=1)
    hx_f = torch.narrow(hx, dim=0, start=0, length=1)

    r_s0_l0f = F.sigmoid(weight_ih_r_l0f @ x_s0f.T + weight_hh_r_l0f @ hx_f.T)
    z_s0_l0f = F.sigmoid(weight_ih_z_l0f @ x_s0f.T + weight_hh_z_l0f @ hx_f.T)
    n_s0_l0f = F.tanh(
        weight_ih_n_l0f @ x_s0f.T + r_s0_l0f * (weight_hh_n_l0f @ hx_f.T)
    )
    h_s1_l0f = ((1.0 - z_s0_l0f) * n_s0_l0f + z_s0_l0f * hx_f.T).T

    # Second step forward
    x_s1f = torch.narrow(x, dim=0, start=1, length=1)

    r_s1_l0f = F.sigmoid(
        weight_ih_r_l0f @ x_s1f.T + weight_hh_r_l0f @ h_s1_l0f.T
    ) 
    z_s1_l0f = F.sigmoid(
        weight_ih_z_l0f @ x_s1f.T + weight_hh_z_l0f @ h_s1_l0f.T
    )
    n_s1_l0f = F.tanh(
        weight_ih_n_l0f @ x_s1f.T + r_s1_l0f * (weight_hh_n_l0f @ h_s1_l0f.T)
    )
    h_s2_l0f = ((1.0 - z_s1_l0f) * n_s1_l0f + z_s1_l0f * h_s1_l0f.T).T

    # Backward weights (these are currently not documented in PyTorch docs)
    # (hidden_size, d * input_size)
    weight_ih_r_l0b = net.weight_ih_l0_reverse[0:hidden_size, 0:1]
    weight_ih_z_l0b = net.weight_ih_l0_reverse[
        hidden_size:(2 * hidden_size), 0:1
    ]
    weight_ih_n_l0b = net.weight_ih_l0_reverse[
        (2 * hidden_size):(3 * hidden_size), 0:1
    ]
    weight_hh_r_l0b = net.weight_hh_l0_reverse[0:hidden_size, 0:1]
    weight_hh_z_l0b = net.weight_hh_l0_reverse[
        hidden_size:(2 * hidden_size), 0:1
    ]
    weight_hh_n_l0b = net.weight_hh_l0_reverse[
        (2 * hidden_size):(3 * hidden_size), 0:1
    ]

    # First step backward
    x_s0b = torch.narrow(x.flip(dims=(0,)), dim=0, start=0, length=1)
    hx_b = torch.narrow(hx.flip(dims=(0,)), dim=0, start=0, length=1)

    r_s0_l0b = F.sigmoid(weight_ih_r_l0b @ x_s0b.T + weight_hh_r_l0b @ hx_b.T)
    z_s0_l0b = F.sigmoid(weight_ih_z_l0b @ x_s0b.T + weight_hh_z_l0b @ hx_b.T)
    n_s0_l0b = F.tanh(
        weight_ih_n_l0b @ x_s0b.T + r_s0_l0b * (weight_hh_n_l0b @ hx_b.T)
    )
    h_s1_l0b = ((1.0 - z_s0_l0b) * n_s0_l0b + z_s0_l0b * hx_b.T).T

    # # Second step backward
    x_s1b = torch.narrow(x.flip(dims=(0,)), dim=0, start=1, length=1)

    r_s1_l0b = F.sigmoid(
        weight_ih_r_l0b @ x_s1b.T + weight_hh_r_l0b @ h_s1_l0b.T
    )
    z_s1_l0b = F.sigmoid(
        weight_ih_z_l0b @ x_s1b.T + weight_hh_z_l0b @ h_s1_l0b.T
    )
    n_s1_l0b = F.tanh(
        weight_ih_n_l0b @ x_s1b.T + r_s1_l0b * (weight_hh_n_l0b @ h_s1_l0b.T)
    )
    h_s2_l0b = ((1.0 - z_s1_l0b) * n_s1_l0b + z_s1_l0b * h_s1_l0b.T).T

    # Concat forward output
    ...

    # Concat backward output
    ...

    # Concat both outputs
    ...
