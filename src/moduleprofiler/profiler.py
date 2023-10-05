import torch
import torch.nn as nn
from typing import Any, Callable, Optional, Union
from .utils import make_list
from .logger import Logger


class ModuleProfiler:
    def __init__(self,
                 input_shape_attr: str = "__input_shape__",
                 output_shape_attr: str = "__output_shape__",
                 ops_attr: str = "__ops__",
                 inference_start_attr: str = "__inference_start__",
                 inference_end_attr: str = "__inference_end__",
                 io_shapes_fn_map: Optional[dict] = None,
                 ops_fn_map: Optional[dict] = None,
                 ts_fmt: str = "%Y-%m-%d %H:%M:%S",
                 verbose: bool = True):
        # TODO:
        # Keep track of ops ('exp', 'sum', 'mul', 'add', 'div', 'diff', etc)
        super().__init__()

        # Params
        self.input_shape_attr = input_shape_attr
        self.output_shape_attr = output_shape_attr
        self.ops_attr = ops_attr
        self.inference_start_attr = inference_start_attr
        self.inference_end_attr = inference_end_attr
        self.io_shapes_fn_map = io_shapes_fn_map
        self.ops_fn_map = ops_fn_map
        self.verbose = verbose
        self._logger = Logger(ts_fmt=ts_fmt)
        self._hook_handles = []

    def _setattr(self, module: nn.Module, attr: Union[str, list],
                 value: Any = None) -> None:
        """ Sets attributes with a value. This is internally used to store
        temporary results in different nested ``nn.Module`` instances.

        Args:
            module (nn.Module): Input module.
            attr (Union[str, list]): Attribute name(s).
            value (Any): Default value of the attribute(s).
        """
        attrs = make_list(attr)

        for attr_ in attrs:
            if hasattr(module, attr_):
                raise ValueError(
                    f"Attribute '{attr_}' already defined in module "
                    f"'{module.__class__.__name__}'. Running "
                    f"'{self.__class__.__name__}' can affect your code. Please"
                    f" rename '{attr_}' to avoid name collisions"
                )

            else:
                setattr(module, attr_, value)

    def _delattr(self, module: nn.Module, attr: Union[str, list]) -> None:
        """ Removes model attribute(s).

        Args:
            module (nn.Module): Input modulei
            attr (Union[str, list]): Name of the attribute(s) to be removed.
        """
        attrs = make_list(attr)

        for attr_ in attrs:
            if hasattr(module, attr_):
                delattr(module, attr_)

    def _dtype_bits(self, dtype: torch.dtype) -> int:
        # Basic dtypes: https://pytorch.org/docs/stable/type_info.html
        # All dtypes: https://pytorch.org/docs/stable/tensor_attributes.html
        if dtype in [torch.uint8, torch.int8, torch.int16, torch.int32,
                     torch.int64]:
            bits = torch.iinfo(dtype).bits

        elif dtype in [torch.float16, torch.bfloat16, torch.float32,
                       torch.float64]:
            bits = torch.finfo(dtype).bits

        else:
            raise AssertionError

        return bits

    def _register_forward_hook(self, module: nn.Module,
                               hook: Callable) -> None:
        self._hook_handles.append(module.register_forward_hook(hook))

    def _register_forward_pre_hook(self, module: nn.Module,
                                   hook: Callable) -> None:
        self._hook_handles.append(module.register_forward_pre_hook(hook))
