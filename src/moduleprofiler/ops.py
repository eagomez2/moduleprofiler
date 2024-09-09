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


def _convtransposend_filter_addition_ops(
        module: nn.ConvTranspose1d,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # Get input filled with ones
    x_ones = torch.ones_like(input[0])

    # Get copy of input modules but with weight filled with ones
    convtranspose1d_ones = type(module)(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        padding_mode=module.padding_mode,
        dilation=module.dilation,
        groups=module.groups,
        bias=False
    )
    torch.nn.init.ones_(convtranspose1d_ones.weight)

    # Compute additions pattern
    total_addition_ops = convtranspose1d_ones(x_ones) - 1.0
    total_addition_ops = torch.sum(total_addition_ops)

    return int(total_addition_ops)


def _convtranspose1d_ops_fn(
        module: nn.ConvTranspose1d,
        input: Tuple[torch.Tensor],
        output: torch.Tensor        
) -> int:
    # Get iput
    x0 = input[0]

    # Get batch size
    batch_size = 1 if x0.ndim == 1 else x0.size(0)

    # Get addition ops
    total_addition_ops = _convtransposend_filter_addition_ops(
        module,
        input,
        output
    )

    total_ops = (
        batch_size
        * ((module.in_channels * module.out_channels) / module.groups)
        * (output.size(-1) * (module.kernel_size[0] + 1) + total_addition_ops)
    )

    # Add bias correction
    if module.bias is None:
        total_ops -= batch_size * module.out_channels * output.size(-1)
    
    return int(total_ops)


def _convtranspose2d_ops_fn(
        module: nn.ConvTranspose2d,
        input: Tuple[torch.Tensor],
        output: torch.Tensor        
) -> int:
    # Get iput
    x0 = input[0]

    # Get batch size
    batch_size = 1 if x0.ndim == 2 else x0.size(0)

    # Get addition ops
    total_addition_ops = _convtransposend_filter_addition_ops(
        module,
        input,
        output
    )

    total_ops = (
        batch_size
        * ((module.in_channels * module.out_channels) / module.groups)
        * (
            output.size(-1) * output.size(-2)
            * (module.kernel_size[0] * module.kernel_size[1] + 1)
            + total_addition_ops
        )
    )

    # Add bias correction
    if module.bias is None:
        total_ops -= (
            batch_size
            * module.out_channels * output.size(-1) * output.size(-2)
        )
    
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
    if module.proj_size != 0:
        raise NotImplementedError("proj_size != 0 is not currently supported")
    
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
                16 * seq_len * batch_size * hidden_size
                * (
                    input_size
                    + (3 * num_layers - 2) * hidden_size
                    + 3.875 * num_layers
                )
            )
        
        else:
            total_ops = (
                8 * seq_len * batch_size * hidden_size
                * (
                    input_size
                    + (2 * num_layers - 1) * hidden_size
                    + 3.875 * num_layers
                )
            )
    
    else:
        if module.bidirectional:
            total_ops = (
                16 * seq_len * batch_size * hidden_size
                * (
                    input_size
                    + (3 * num_layers - 2) * hidden_size
                    + 2.875 * num_layers
                )
            )
        
        else:
            total_ops = (
                8 * seq_len * batch_size * hidden_size
                * (
                    input_size
                    + (2 * num_layers - 1) * hidden_size
                    + 2.875 * num_layers
                )
            )

    return int(total_ops)


def _relu_ops_fn(
        module: nn.ReLU,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # NOTE: This estimation is not straightforward since the max() function is
    # used. A simple estimation is to assume a single operation per element.
    return input[0].numel()


def _leakyrelu_ops_fn(
        module: nn.PReLU,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    return input[0].numel() * 4


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


def _maxpool1d_ops_fn(
        module: nn.MaxPool1d,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    batch_size = 1 if input[0].ndim == 2 else input[0].size(0)
    in_channels = input[0].size(-2)
    return batch_size * in_channels * output.size(-1)


def _maxpool2d_ops_fn(
        module: nn.MaxPool2d,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    batch_size = 1 if input[0].ndim == 3 else input[0].size(0)
    in_channels = input[0].size(-3)
    return batch_size * in_channels * output.size(-2) * output.size(-1)


def _selu_ops_fn(
        module: nn.SELU,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    return input[0].numel() * 7


def _softplus_ops_fn(
        module: nn.Softplus,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    return input[0].numel() * 5


def _layernorm_ops_fn(
        module: nn.LayerNorm,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    if not module.elementwise_affine:
        num_elements = (
            module.normalized_shape if isinstance(module.normalized_shape, int)
            else math.prod(module.normalized_shape)
        )
        total_ops = 5 * num_elements + 3
    
    else:
        if module.bias is not None:
            total_ops = 7 * num_elements + 3
        
        else:
            total_ops = 6 * num_elements + 3
    
    # Add batch size
    total_ops *= input[0].size(0)
        
    return total_ops


def _avgpool1d_ops_fn(
        module: nn.AvgPool1d,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    batch_size = 1 if input[0].ndim == 2 else input[0].size(0)
    num_channels = input[0].size(-2)

    return batch_size * num_channels * output.size(-1) * module.kernel_size[0]


def _avgpool2d_ops_fn(
        module: nn.AvgPool2d,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    batch_size = 1 if input[0].ndim == 3 else input[0].size(0)
    num_channels = input[0].size(-3)

    return (
        batch_size * num_channels
        * output.size(-1) * output.size(-2)
        * module.kernel_size[0] * module.kernel_size[1]
    )


def get_default_ops_map() -> dict:
    return {
        # Default method
        "default": _default_ops_fn,

        # Layers
        nn.Identity: _identity_ops_fn,
        nn.Linear: _linear_ops_fn,
        nn.Conv1d: _conv1d_ops_fn,
        nn.Conv2d: _conv2d_ops_fn,
        nn.ConvTranspose1d: _convtranspose1d_ops_fn,
        nn.ConvTranspose2d: _convtranspose2d_ops_fn,
        nn.GRUCell: _grucell_ops_fn,
        nn.GRU: _gru_ops_fn,
        nn.LSTMCell: _lstmcell_ops_fn,
        nn.LSTM: _lstm_ops_fn,

        # Norm
        nn.LayerNorm: _layernorm_ops_fn,

        # Pooling
        nn.AvgPool1d: _avgpool1d_ops_fn,
        nn.AvgPool2d: _avgpool2d_ops_fn,
        nn.AdaptiveMaxPool1d: _maxpool1d_ops_fn,
        nn.AdaptiveAvgPool2d: _maxpool2d_ops_fn,
        nn.MaxPool1d: _maxpool1d_ops_fn,
        nn.MaxPool2d: _maxpool2d_ops_fn,

        # Activations
        nn.ReLU: _relu_ops_fn,
        nn.LeakyReLU: _leakyrelu_ops_fn,
        nn.ELU: _elu_ops_fn,
        nn.PReLU: _prelu_ops_fn,
        nn.SELU: _selu_ops_fn,
        nn.Sigmoid: _sigmoid_ops_fn,
        nn.Softmax: _softmax_ops_fn,
        nn.Softplus: _softplus_ops_fn,
        nn.Tanh: _tanh_ops_fn
    }
