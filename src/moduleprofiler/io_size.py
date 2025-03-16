import torch
import torch.nn as nn
from typing import (
    Any,
    Tuple
)


def _get_item_repr(item: Any) -> Any:
    # None
    if item is None:
        item_repr = None
    
    # Single tensor
    elif isinstance(item, torch.Tensor):
        item_repr = tuple(item.size())
    
    # List of tensors and possibly some other types
    elif isinstance(item, list):
        item_repr = [
            tuple(i.size()) if isinstance(i, torch.Tensor)
            else type(i).__name__
            for i in item
        ]
    
    # Tuple of tensors and possibly some other types
    elif isinstance(item, tuple):
        item_repr = [
            tuple(i.size()) if isinstance(i, torch.Tensor)
            else type(i).__name__
            for i in item
        ]
        item_repr = tuple(item_repr)
    
    # Set of tensors and possibly some other types
    elif isinstance(item, set):
        item_repr = [
            tuple(i.size()) if isinstance(i, torch.Tensor)
            else type(i).__name__
            for i in item
        ]
        item_repr = set(item_repr) 
    
    # Dict of tensors and possibly some other types
    elif isinstance(item, dict):
        item_repr = {k: _get_item_repr(v) for k, v in item.items()}
    
    else:
        raise NotImplementedError
    
    return item_repr


def _default_io_size_fn(
        module: nn.Module,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> Tuple[tuple]:
    # input is None
    if input is None:
        input_ = None

    # input is a single tensor
    elif len(input) == 1 and isinstance(input[0], torch.Tensor):
        input_ = tuple(input[0].size())
    
    else:
        input_ = tuple(_get_item_repr(i) for i in input)
    
    # Get output size
    if output is None:
        output_ = None
    
    elif isinstance(output, torch.Tensor):
        output_ = tuple(output.size())
    
    elif isinstance(output, dict):
        output_ = _get_item_repr(output)
    
    elif isinstance(output, list):
        output_ = [_get_item_repr(o) for o in output]
    
    elif isinstance(output, set):
        output_ = {_get_item_repr(o) for o in output}
    
    elif isinstance(output, tuple):
        output_ = tuple(_get_item_repr(o) for o in output)
    
    else:
        raise NotImplementedError

    return input_, output_


def _gru_io_size_fn(
        module: nn.GRU,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> Tuple[tuple]:
    input_shape = input[0].size()
    output_shape = output[0][0].size()
    hidden_state_shape = output[1][0].size()

    return (
        tuple(input_shape),
        (tuple(output_shape), tuple(hidden_state_shape))
    )


def _lstm_io_size_fn(
        module: nn.LSTM,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> Tuple[tuple]:
    input_shape = input[0].size()
    output_shape = output[0][0].size()
    hidden_state_shape = output[1][0].size()
    cell_state_shape = output[1][1].size()

    return (
        tuple(input_shape),
        (
            tuple(output_shape),
            (tuple(hidden_state_shape), tuple(cell_state_shape))
        )
    )


def get_default_io_size_map() -> dict:
    return {
        nn.GRUCell: _gru_io_size_fn,
        nn.GRU: _gru_io_size_fn,
        nn.LSTMCell: _lstm_io_size_fn,
        nn.LSTM: _lstm_io_size_fn,
        "default": _default_io_size_fn
    }
