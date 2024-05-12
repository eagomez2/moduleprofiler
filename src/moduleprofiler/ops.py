import torch
import math
import torch.nn as nn
from typing import (
    Any,
    Tuple
)


def _default_ops_fn(
        module: nn.Module,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> Any:
    return None


def _identity_ops_fn(
        module: nn.Identity,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    return 0


def _linear_ops_fn(
        module: nn.Linear,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # Get input batch_size and num_channels (if any)
    num_seqs = input[0].size()[:-1].numel()

    # Get ops per sequence
    if module.bias is not None:
        ops_per_seq = 2.0 * module.out_features * module.in_features

    else:
        ops_per_seq = module.out_features * (2.0 * module.in_features - 1.0)

    total_ops = num_seqs * ops_per_seq

    return int(total_ops)


def _conv1d_ops_fn(
        module: nn.Conv1d,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # Get input
    x0 = input[0]

    # Get batch size
    batch_size = 1 if len(x0.size()) == 2 else x0.size(0)

    # Compute input length of a single channel
    x0_len = x0.size(-1)

    # Avoid invalid operations caused by incompatible types
    padding = module.padding[0]
    dilation = module.dilation[0]
    kernel_size = module.kernel_size[0]
    stride = module.stride[0]

    # Compute output length
    # NOTE: It can also be directly derived from the module's output
    y0_len = (
        ((x0_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride)
        + 1
    )

    # Compute number of filters
    num_filters = ((module.in_channels * module.out_channels) / module.groups)

    # Compute operations per filter
    ops_per_filter = y0_len * (2 * kernel_size - 1)

    # Compute number of aggregation operations
    aggr_ops_bias = (0 if module.bias is not None else 1)
    aggr_ops = (
        y0_len
        * module.out_channels
        * ((module.in_channels / module.groups) - aggr_ops_bias)
    )

    total_ops = batch_size * (num_filters * ops_per_filter + aggr_ops)

    return int(total_ops)


def _grucell_ops_fn(
        module: nn.GRUCell,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # Get params
    batch_size = 1 if len(input[0].size()) == 1 else input[0].size(0)
    h_out = module.hidden_size
    h_in = module.input_size

    if module.bias is not None:
        r_ops = 2 * batch_size * h_out * (h_in + h_out + 2)
        z_ops = r_ops
        n_ops = batch_size * h_out * (9 + 2 * (h_in + h_out))
    
    else:
        r_ops = 2 * batch_size * h_out * (h_in + h_out + 1)
        z_ops = r_ops
        n_ops = batch_size * h_out * (9 + 2 * (h_in + h_out - 1))

    # Same regardless of bias
    h_prime_ops = 4 * batch_size * h_out

    total_ops = r_ops + z_ops + n_ops + h_prime_ops

    return total_ops


def _gru_ops_fn(
        module: nn.GRU,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # Get params
    if len(input[0].size()) == 2:
        batch_size = 1
        num_steps = input[0].size(0)
    
    elif input[0].size() == 3:
        batch_size = (
            input[0].size(0) if module.batch_first else input[0].size(1)
        )
        num_steps = (
            input[0].size(1) if module.batch_first else input[0].size(0)
        )
    
    num_layers = module.num_layers
    num_directions = 2 if module.bidirectional else 1
    h_out = module.hidden_size
    h_in = module.input_size

    if module.bias is not None:
        r_ops = 2 * batch_size * h_out * (h_in + h_out + 2)
        z_ops = r_ops
        n_ops = batch_size * h_out * (9 + 2 * (h_in + h_out))
    
    else:
        r_ops = 2 * batch_size * h_out * (h_in + h_out + 1)
        z_ops = r_ops
        n_ops = batch_size * h_out * (9 + 2 * (h_in + h_out - 1))

    # Same regardless of bias
    h_prime_ops = 4 * batch_size * h_out

    grucell_ops = r_ops + z_ops + n_ops + h_prime_ops
    total_ops = num_directions * num_steps * num_layers * grucell_ops

    return total_ops


def _lstmcell_ops_fn(
        module: nn.LSTMCell,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # Get params
    batch_size = 1 if len(input[0].size()) == 1 else input[0].size(0)
    h_out = module.hidden_size
    h_in = module.input_size

    if module.bias is not None:
        i_ops = 2 * batch_size * h_out * (2 + h_in + h_out)
        g_ops = 2 * batch_size * h_out * (4 + h_in + h_out)
    
    else:
        i_ops = 2 * batch_size * h_out * (1 + h_in + h_out)
        g_ops = 2 * batch_size * h_out * (3 + h_in + h_out)
    
    # Other gate ops
    f_ops = i_ops
    g_ops = i_ops
    c_prime_ops = 3 * batch_size * h_out
    h_prime_ops = 8 * batch_size * h_out

    total_ops = i_ops + g_ops + f_ops + c_prime_ops + h_prime_ops

    return total_ops


def _lstm_ops_fn(
        module: nn.LSTM,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # NOTE: Not currently implemented!
    assert module.proj_size == 0

    # Get params
    if len(input[0].size()) == 2:
        batch_size = 1
        num_steps = input[0].size(0)
    
    elif input[0].size() == 3:
        batch_size = (
            input[0].size(0) if module.batch_first else input[0].size(1)
        )
        num_steps = (
            input[0].size(1) if module.batch_first else input[0].size(0)
        )
    
    num_layers = module.num_layers
    num_directions = 2 if module.bidirectional else 1
    h_out = module.hidden_size
    h_in = module.input_size

    if module.bias is not None:
        i_ops = 2 * batch_size * h_out * (2 + h_in + h_out)
        g_ops = 2 * batch_size * h_out * (4 + h_in + h_out)
    
    else:
        i_ops = 2 * batch_size * h_out * (1 + h_in + h_out)
        g_ops = 2 * batch_size * h_out * (3 + h_in + h_out)
    
    # Other gate ops
    f_ops = i_ops
    g_ops = i_ops
    c_prime_ops = 3 * batch_size * h_out
    h_prime_ops = 8 * batch_size * h_out

    lstmcell_ops = i_ops + g_ops + f_ops + c_prime_ops + h_prime_ops
    total_ops = num_directions * num_steps * num_layers * lstmcell_ops

    return total_ops


def _relu_ops_fn(
        module: nn.ReLU,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # NOTE: This estimation is not straightforward since the max() function is
    # used. A simple estimation is to assume a single operation per element.
    return input[0].numel()


def _sigmoid_ops_fn(
        module: nn.Sigmoid,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # NOTE: Exponential is considered as a single op
    return input[0].numel() * 3


def _softmax_ops_fn(
        module: nn.Softmax,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # Ops per row along dim dimension
    # NOTE: Exponential is considered as a single op
    row_ops = 4 * input[0].size(module.dim) - 1

    # Get remaining dims
    if len(input[0].size()) == 1:
        other_dims = 1
    
    else:
        other_dims = list(input[0].size())
        other_dims.pop(module.dim)
        other_dims = math.prod(other_dims)
    
    total_ops = other_dims * row_ops

    return total_ops


def _elu_ops_fn(
        module: nn.ELU,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # NOTE: Exponential is considered as a single op
    return input[0].numel() * 3


def _prelu_ops_fn(
        module: nn.PReLU,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    return input[0].numel() * 4


def _tanh_ops_fn(
        module: nn.Tanh,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    return input[0].numel() * 7


def _get_default_ops_map() -> dict:
    return {
        # Default method
        "default": _default_ops_fn,

        # Layers
        nn.Identity: _identity_ops_fn,
        nn.Linear: _linear_ops_fn,
        nn.Conv1d: _conv1d_ops_fn,
        nn.GRUCell: _grucell_ops_fn,
        nn.GRU: _gru_ops_fn,
        nn.LSTMCell: _lstmcell_ops_fn,
        nn.LSTM: _lstm_ops_fn,

        # Norm

        # Pooling

        # Activations
        nn.ReLU: _relu_ops_fn,
        nn.ELU: _elu_ops_fn,
        nn.PReLU: _prelu_ops_fn,
        nn.Sigmoid: _sigmoid_ops_fn,
        nn.Softmax: _softmax_ops_fn,
        nn.Tanh: _tanh_ops_fn
    }
