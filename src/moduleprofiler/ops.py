import torch
import torch.nn as nn
from typing import Any, Tuple


def _default_ops_fn(module: nn.Module, input: Tuple[torch.Tensor],
                    output: torch.Tensor) -> Any:
    return None


_DEFAULT_OPS_MAP = {
    "default": _default_ops_fn
}
