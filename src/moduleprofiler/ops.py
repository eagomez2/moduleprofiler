import torch
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
    # NOTE: Exponential is considered as a single op here
    return input[0].numel() * 3


def _softmax_ops_fn(
        module: nn.Softmax,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    ...


_DEFAULT_OPS_MAP = {
    # Default method
    "default": _default_ops_fn,

    # Layers
    nn.Identity: _identity_ops_fn,
    nn.Linear: _linear_ops_fn,
    nn.Conv1d: _conv1d_ops_fn,

    # Activations
    nn.ReLU: _relu_ops_fn,
    nn.Sigmoid: _sigmoid_ops_fn,
    nn.Softmax: _softmax_ops_fn
}
