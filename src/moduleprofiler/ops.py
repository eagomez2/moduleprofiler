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
    batch_size = 1 if x0.ndim == 2 else x0.size(0)

    # NOTE: kernel_size[0] is used to avoid issues with invalid data types
    if module.bias is not None:
        numerator = (
            module.out_channels * output.size(-1)
            * module.in_channels * 2 * module.kernel_size[0]
        )
    
    else:
        numerator = (
            module.out_channels * output.size(-1)
            * (module.in_channels * 2 * module.kernel_size[0] - module.groups)
        )
    
    total_ops = batch_size * (numerator / module.groups)

    return int(total_ops)


def _conv2d_ops_fn(
        module: nn.Conv2d,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # Get input
    x0 = input[0]

    # Get batch size
    batch_size = 1 if x0.ndim == 2 else x0.size(0)

    # Avoid invalid operations caused by incompatible types
    kernel_size_prod = (
        module.kernel_size[0] if isinstance(module.kernel_size, int)
        else module.kernel_size[0] * module.kernel_size[1]
    )

    if module.bias is not None:
        numerator = (
            module.out_channels * output.size(-1) * output.size(-2)
            * module.in_channels * 2 * kernel_size_prod
        )
    
    else:
        numerator = (
            module.out_channels * output.size(-1) * output.size(-2)
            * (module.in_channels * 2 * kernel_size_prod - module.groups)
        )
    
    total_ops = batch_size * (numerator / module.groups)

    return int(total_ops)


def _grucell_ops_fn(
        module: nn.GRUCell,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    batch_size = 1 if len(input[0].size()) == 1 else input[0].size(0)
    
    if module.bias is not None:
        total_ops = (
            6 * batch_size * output.size(-1)
            * (input[0].size(-1) + output.size(-1) + 3.5)
        )
    
    else:
        total_ops = (
            6 * batch_size * output.size(-1)
            * (input[0].size(-1) + output.size(-1) + 2.5)
        )

    return total_ops


def _gru_ops_fn(
        module: nn.GRU,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # Get params
    if input[0].ndim == 2:
        batch_size = 1
        seq_len = input[0].size(0)
    
    elif input[0].ndim == 3:
        batch_size = (
            input[0].size(0) if module.batch_first else input[0].size(1)
        )
        seq_len = (
            input[0].size(1) if module.batch_first else input[0].size(0)
        )
    
    input_size = module.input_size
    hidden_size = module.hidden_size
    num_layers = module.num_layers

    if module.bias:
        if module.bidirectional:
            total_ops = (
                12 * seq_len * batch_size * hidden_size
                * (
                    input_size
                    + (3 * num_layers - 2) * hidden_size
                    + 3.5 * num_layers
                )
            )
        
        else:
            total_ops = (
                6 * seq_len * batch_size * hidden_size
                * (
                    input_size
                    + (2 * num_layers - 1) * hidden_size
                    + 3.5 * num_layers
                )
            )
    
    else:
        if module.bidirectional:
            total_ops = (
                12 * seq_len * batch_size * hidden_size
                * (
                    input_size
                    + (3 * num_layers - 2) * hidden_size
                    + 2.5 * num_layers
                )
            )
        
        else:
            total_ops = (
                6 * seq_len * batch_size * hidden_size
                * (
                    input_size
                    + (2 * num_layers - 1) * hidden_size
                    + 2.5 * num_layers
                )
            )

    return int(total_ops)


def _lstmcell_ops_fn(
        module: nn.LSTMCell,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    batch_size = 1 if len(input[0].size()) == 1 else input[0].size(0)
    
    if module.bias is not None:
        total_ops = (
            8 * batch_size * output.size(-1)
            * (input[0].size(-1) + output.size(-1) + 3.875)
        )
    
    else:
        total_ops = (
            8 * batch_size * output.size(-1)
            * (input[0].size(-1) + output.size(-1) + 2.875)
        )

    return int(total_ops)


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
        nn.Conv2d: _conv2d_ops_fn,
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
