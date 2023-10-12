import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Any, Callable, Dict, Optional, Tuple, Union
from .utils import make_list, dict_merge
from .logger import Logger
from .io_shapes import _DEFAULT_IO_SHAPES_FN_MAP
from .ops import _DEFAULT_OPS_MAP


class ModuleProfiler:
    def __init__(self,
                 input_shape_attr: str = "__input_shape__",
                 output_shape_attr: str = "__output_shape__",
                 ops_attr: str = "__ops__",
                 inference_start_attr: str = "__inference_start__",
                 inference_end_attr: str = "__inference_end__",
                 io_shapes_fn_map: dict = _DEFAULT_IO_SHAPES_FN_MAP,
                 ops_fn_map: dict = _DEFAULT_OPS_MAP,
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
        """ Returns the size in bits of a numeric data type.

        Args:
            dtype (torch.dtype): Data type.

        Returns:
            (int): Size of ``dtype`` in bits.
        """
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
        """ Registers a forward hook in a module and stores the corresponding
        handle to be deleted later.

        Args:
            module (nn.Module): Input module.
            hook (Callable): Hook to be registered.
        """
        self._hook_handles.append(module.register_forward_hook(hook))

    def _register_forward_pre_hook(self, module: nn.Module,
                                   hook: Callable) -> None:
        """ Registers a forward pre hook in a module and stored the
        corresponding handle to be deleted later.

        Args:
            module (nn.Module): Input module.
            hook (Callable): Hook to be registered.
        """
        self._hook_handles.append(module.register_forward_pre_hook(hook))

    def _merge_specs(self, specs: Tuple[Dict[str, dict]]) -> Dict[str, dict]:
        """ Merges two or more ``dict`` instances containing the same keys.
        This will result in a ``dict`` containing the values of both ``dict``
        instances.

        Args:
            specs (Tuple[Dict[str, dict]]): Specifications ``dict`` or tuple
                with two or more specification ``dict`` instances to be
                merged.

        Returns:
            (Dict[str, dict]): Merged ``dict`` containing the same keys
            but a merged set of values for each key.
        """
        # Take first dict as reference
        ref_spec = specs[0]
        other_specs = specs[1:]

        # Check all specs have the same keys
        for idx, spec in enumerate(other_specs):
            if not spec.keys() == ref_spec.keys():
                ref_spec_keys_repr =\
                    ", ".join([f"'{k}'" for k in ref_spec.keys()])
                spec_keys_repr =\
                    ", ".join([f"'{k}'" for k in spec.keys()])

                raise ValueError(
                    f"All specs to be merged should have the same keys. Found "
                    f"specs[0] and specs[{idx + 1}] have different keys: "
                    f"specs[0].keys()={ref_spec_keys_repr} and "
                    f"specs[{idx + 1}].keys()={spec_keys_repr}"
                )

        # Merge dicts
        merged_specs = ref_spec

        for spec in specs[1:]:
            for k in merged_specs.keys():
                merged_specs[k] = dict_merge(merged_specs[k], spec[k])

        return merged_specs

    def count_params(self, module: nn.Module, param_size: bool = True,
                     param_dtype: bool = True, percent: bool = True) -> dict:
        """ Counts the number of parameters in a model.

        Args:
            module (nn.Module): Model whose parameters will be counted.
            param_size (bool): If ``True``, the size in bits of each parameters
                will be calculated.
            param_dtype (bool): If ``True``, the data type of different
                parameters will be reported.
            percent (bool): If ``True``, the percentage each parameter
                represents with respect to the total amount of parameters of
                the model will be reported.

        Returns:
            (dict): Analysis results.
        """
        data = {}

        if self.verbose:
            self._logger.log(
                "Counting parameters of "
                f"<b><magenta>{module.__class__.__name__}</magenta></b>"
            )

        # TODO: Add progress bar if verbose=True, or use disable=True if it
        # should not be displayed. Also, consider adding tqdm to the Logger
        # class since it is required to print while the progress bar is on
        for idx, (n, m) in enumerate(module.named_modules()):
            # First entry corresponds to the module itself
            if idx == 0:
                n = "__root__"

            data[n] = {
                "type": m.__class__.__name__,
                "trainable_params": 0,
                "nontrainable_params": 0
            }

            for p in m.parameters():
                if p.requires_grad:
                    data[n]["trainable_params"] += p.numel()

                    if param_dtype:
                        data[n]["trainable_params_dtype"] = p.dtype

                    if param_size:
                        dtype_size = self._dtype_bits(p.dtype)
                        data[n]["trainable_params_size_bits"] = \
                            data[n]["trainable_params"] * dtype_size

                else:
                    data[n]["nontrainable_params"] += p.numel()

                    if param_dtype:
                        data[n]["nontrainable_params_dtype"] = p.dtype

                    if param_size:
                        dtype_size = self._dtype_bits(p.dtype)
                        data[n]["nontrainable_params_size_bits"] = \
                            data[n]["nontrainable_params"] * dtype_size

        if percent:
            # Percentage of total params
            total_params = data["__root__"]["trainable_params"] + \
                           data["__root__"]["nontrainable_params"]

            for k in data.keys():
                data[k]["trainable_params_percent"] = (
                    data[k]["trainable_params"] / total_params
                )

                data[k]["nontrainable_params_percent"] = (
                    data[k]["nontrainable_params"] / total_params
                )

        return data
