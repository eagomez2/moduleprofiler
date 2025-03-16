import pytest
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, forward_fn: str) -> None:
        super().__init__()

        # Params
        match forward_fn:
            case "tensor":
                self._forward_fn = self._forward_tensor
            
            case "tuple":
                ...
            
            case "dict":
                ...

            case _:
                raise AssertionError

    def _forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self._forward_fn(*args, **kwargs)
