import torch
import torch.nn as nn
from typing import Any, Tuple


def _default_ops_fn(module: nn.Module, input: Tuple[torch.Tensor],
                    output: torch.Tensor) -> Any:
    return None

def _conv1d_ops_fn(module: nn.Conv1d, input: Tuple[torch.Tensor],
                   output: torch.Tensor) -> int:
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


_DEFAULT_OPS_MAP = {
    "default": _default_ops_fn,
    nn.Conv1d: _conv1d_ops_fn
}
