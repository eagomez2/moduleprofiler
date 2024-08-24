import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


@pytest.mark.parametrize(
        "batch_size, input_size, hidden_size", [
            # batch_size=1, bias=False
            (1, 2, 2),
            (1, 3, 6),
            (1, 4, 8),

            # batch_size=2
            (2, 2, 2),
            (2, 3, 6),
            (2, 4, 8),
        ]
)
def test_grucell_output_match_no_bias(
    batch_size: int,
    input_size: int,
    hidden_size: int,
) -> None:
    # Sample input and hidden state tensors
    x = torch.rand((batch_size, input_size), dtype=torch.float32)
    hx = torch.rand((batch_size, hidden_size), dtype=torch.float32)

    # Built-in module
    net = nn.GRUCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=False
    )

    # Automatic calculation
    h_prime_torch = net(x, hx)

    # Manual calculation (weight order ir, iz, in)
    # (hidden_size, input_size)
    weight_ih_r = net.weight_ih[0:hidden_size, ...]
    weight_ih_z = net.weight_ih[hidden_size:(2 * hidden_size), ...]
    weight_ih_n = net.weight_ih[(2 * hidden_size):(3 * hidden_size), ...]

    # Weight order hr, hz, hn)
    # (hidden_size, hidden_size)
    weight_hh_r = net.weight_hh[0:hidden_size, ...]
    weight_hh_z = net.weight_hh[hidden_size:(2 * hidden_size), ...]
    weight_hh_n = net.weight_hh[(2 * hidden_size):(3 * hidden_size), ...]
    
    # Perform forward pass
    r = F.sigmoid(weight_ih_r @ x.T + weight_hh_r @ hx.T)
    z = F.sigmoid(weight_ih_z @ x.T + weight_hh_z @ hx.T)
    n = F.tanh(weight_ih_n @ x.T + r * (weight_hh_n @ hx.T))
    h_prime_mp = ((1.0 - z) * n + z * hx.T).T

    # NOTE: Computing all with pytorch function may differ from the underlying
    # optimized module implementation so small differences are expected
    assert torch.allclose(h_prime_torch, h_prime_mp, atol=1e-5)


@pytest.mark.parametrize(
        "batch_size, input_size, hidden_size", [
            # batch_size=1, bias=False
            (1, 2, 2),
            (1, 3, 6),
            (1, 4, 8),

            # batch_size=2
            (2, 2, 2),
            (2, 3, 6),
            (2, 4, 8),
        ]
)
def test_grucell_output_match_bias(
    batch_size: int,
    input_size: int,
    hidden_size: int,
) -> None:
    # Sample input and hidden state tensors
    x = torch.rand((batch_size, input_size), dtype=torch.float32)
    hx = torch.rand((batch_size, hidden_size), dtype=torch.float32)

    # Built-in module
    net = nn.GRUCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True
    )

    # Automatic calculation
    h_prime_torch = net(x, hx)

    # Manual calculation (weight and bias order ir, iz, in)
    # (hidden_size, input_size)
    weight_ih_r = net.weight_ih[0:hidden_size, ...]
    weight_ih_z = net.weight_ih[hidden_size:(2 * hidden_size), ...]
    weight_ih_n = net.weight_ih[(2 * hidden_size):(3 * hidden_size), ...]
    bias_ih_r = net.bias_ih[0:hidden_size]
    bias_ih_z = net.bias_ih[hidden_size:(2 * hidden_size)]
    bias_ih_n = net.bias_ih[(2 * hidden_size):(3 * hidden_size)]

    # Weight and bias order hr, hz, hn)
    # (hidden_size, hidden_size)
    weight_hh_r = net.weight_hh[0:hidden_size, ...]
    weight_hh_z = net.weight_hh[hidden_size:(2 * hidden_size), ...]
    weight_hh_n = net.weight_hh[(2 * hidden_size):(3 * hidden_size), ...]
    bias_hh_r = net.bias_hh[0:hidden_size]
    bias_hh_z = net.bias_hh[hidden_size:(2 * hidden_size)]
    bias_hh_n = net.bias_hh[(2 * hidden_size):(3 * hidden_size)]

    # Perform forward pass
    r = F.sigmoid(
        weight_ih_r @ x.T + bias_ih_r.unsqueeze(1)
        + weight_hh_r @ hx.T + bias_hh_r.unsqueeze(1)
    )
    z = F.sigmoid(
        weight_ih_z @ x.T + bias_ih_z.unsqueeze(1)
        + weight_hh_z @ hx.T + bias_hh_z.unsqueeze(1)
    )
    n = F.tanh(
        weight_ih_n @ x.T + bias_ih_n.unsqueeze(1)
        + r * (weight_hh_n @ hx.T + bias_hh_n.unsqueeze(1))
    )
    h_prime_mp = ((1.0 - z) * n + z * hx.T).T

    # NOTE: Computing all with pytorch function may differ from the underlying
    # optimized module implementation so small differences are expected
    assert torch.allclose(h_prime_torch, h_prime_mp, atol=1e-5)
