import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


@pytest.mark.parametrize(
        "batch_size, input_size, hidden_size", [
            # batch_size=1
            (1, 2, 2),
            (1, 3, 6),
            (1, 4, 8),

            # batch_size=2
            (2, 2, 2),
            (2, 3, 6),
            (2, 4, 8),
        ]
)
def test_lstmcell_output_match_no_bias(
        batch_size: int,
        input_size: int,
        hidden_size: int
) -> None:
    # Sample input and hidden state tensors
    x = torch.rand((batch_size, input_size), dtype=torch.float32)
    hx = torch.rand((batch_size, hidden_size), dtype=torch.float32)
    cx = torch.rand((batch_size, hidden_size), dtype=torch.float32)

    # Built-in module
    net = nn.LSTMCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=False
    )

    # Automatic calculation
    h_prime_torch, c_prime_torch = net(x, (hx, cx))

    # Manual calculation (weight order ii, if, ig, io)
    # (hidden_size, input_size)
    weight_ih_i = net.weight_ih[0:hidden_size, ...]
    weight_ih_f = net.weight_ih[hidden_size:(2 * hidden_size), ...]
    weight_ih_g = net.weight_ih[(2 * hidden_size):(3 * hidden_size), ...]
    weight_ih_o = net.weight_ih[(3 * hidden_size):(4 * hidden_size), ...]

    # Weight order hi, hf, hg, ho
    weight_hh_i = net.weight_hh[0:hidden_size, ...]
    weight_hh_f = net.weight_hh[hidden_size:(2 * hidden_size), ...]
    weight_hh_g = net.weight_hh[(2 * hidden_size):(3 * hidden_size), ...]
    weight_hh_o = net.weight_hh[(3 * hidden_size):(4 * hidden_size), ...]

    # Perform forward pass
    i = F.sigmoid(weight_ih_i @ x.T + weight_hh_i @ hx.T)
    f = F.sigmoid(weight_ih_f @ x.T + weight_hh_f @ hx.T)
    g = F.tanh(weight_ih_g @ x.T + weight_hh_g @ hx.T)
    o = F.sigmoid(weight_ih_o @ x.T + weight_hh_o @ hx.T)
    c_prime_mp = (f * cx.T + i * g).T
    h_prime_mp = (o.T * F.tanh(c_prime_mp))

    # NOTE: Computing all with pytorch function may differ from the underlying
    # optimized module implementation so small differences are expected
    assert (
        torch.allclose(h_prime_torch, h_prime_mp, atol=1e-5)
        and torch.allclose(c_prime_torch, c_prime_mp, atol=1e-5)
    )


@pytest.mark.parametrize(
        "batch_size, input_size, hidden_size", [
            # batch_size=1
            (1, 2, 2),
            (1, 3, 6),
            (1, 4, 8),

            # batch_size=2
            (2, 2, 2),
            (2, 3, 6),
            (2, 4, 8),
        ]
)
def test_lstmcell_output_match_bias(
        batch_size: int,
        input_size: int,
        hidden_size: int
) -> None:
    # Sample input and hidden state tensors
    x = torch.rand((batch_size, input_size), dtype=torch.float32)
    hx = torch.rand((batch_size, hidden_size), dtype=torch.float32)
    cx = torch.rand((batch_size, hidden_size), dtype=torch.float32)

    # Built-in module
    net = nn.LSTMCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True
    )

    # Automatic calculation
    h_prime_torch, c_prime_torch = net(x, (hx, cx))

    # Manual calculation (weight order ii, if, ig, io)
    # (hidden_size, input_size)
    weight_ih_i = net.weight_ih[0:hidden_size, ...]
    weight_ih_f = net.weight_ih[hidden_size:(2 * hidden_size), ...]
    weight_ih_g = net.weight_ih[(2 * hidden_size):(3 * hidden_size), ...]
    weight_ih_o = net.weight_ih[(3 * hidden_size):(4 * hidden_size), ...]
    bias_ih_i = net.bias_ih[0:hidden_size]
    bias_ih_f = net.bias_ih[hidden_size:(2 * hidden_size)]
    bias_ih_g = net.bias_ih[(2 * hidden_size):(3 * hidden_size)]
    bias_ih_o = net.bias_ih[(3 * hidden_size):(4 * hidden_size)]

    # Weight order hi, hf, hg, ho
    weight_hh_i = net.weight_hh[0:hidden_size, ...]
    weight_hh_f = net.weight_hh[hidden_size:(2 * hidden_size), ...]
    weight_hh_g = net.weight_hh[(2 * hidden_size):(3 * hidden_size), ...]
    weight_hh_o = net.weight_hh[(3 * hidden_size):(4 * hidden_size), ...]
    bias_hh_i = net.bias_hh[0:hidden_size]
    bias_hh_f = net.bias_hh[hidden_size:(2 * hidden_size)]
    bias_hh_g = net.bias_hh[(2 * hidden_size):(3 * hidden_size)]
    bias_hh_o = net.bias_hh[(3 * hidden_size):(4 * hidden_size)]

    # Perform forward pass
    i = F.sigmoid(
        weight_ih_i @ x.T + bias_ih_i.unsqueeze(1)
        + weight_hh_i @ hx.T + bias_hh_i.unsqueeze(1)
    )
    f = F.sigmoid(
        weight_ih_f @ x.T + bias_ih_f.unsqueeze(1)
        + weight_hh_f @ hx.T + bias_hh_f.unsqueeze(1)
    )
    g = F.tanh(
        weight_ih_g @ x.T + bias_ih_g.unsqueeze(1)
        + weight_hh_g @ hx.T + bias_hh_g.unsqueeze(1)
    )
    o = F.sigmoid(
        weight_ih_o @ x.T + bias_ih_o.unsqueeze(1)
        + weight_hh_o @ hx.T + bias_hh_o.unsqueeze(1)
    )
    c_prime_mp = (f * cx.T + i * g).T
    h_prime_mp = (o.T * F.tanh(c_prime_mp))

    # NOTE: Computing all with pytorch function may differ from the underlying
    # optimized module implementation so small differences are expected
    assert (
        torch.allclose(h_prime_torch, h_prime_mp, atol=1e-5)
        and torch.allclose(c_prime_torch, c_prime_mp, atol=1e-5)
    )
