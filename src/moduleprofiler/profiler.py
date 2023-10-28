import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from typing import Any, Callable, Dict, Tuple, Union
from time import perf_counter
from .utils import make_list, dict_merge, add_extension, get_hardware_specs
from .logger import Logger
from .io_size import _DEFAULT_IO_SIZE_FN_MAP
from .ops import _DEFAULT_OPS_MAP


# TODO: Separate layers by type when computing metrics?
class ModuleProfiler:
    def __init__(self,
                 input_size_attr: str = "__input_size__",
                 output_size_attr: str = "__output_size__",
                 ops_attr: str = "__ops__",
                 inference_start_attr: str = "__inference_start__",
                 inference_end_attr: str = "__inference_end__",
                 io_size_fn_map: dict = _DEFAULT_IO_SIZE_FN_MAP,
                 ops_fn_map: dict = _DEFAULT_OPS_MAP,
                 ts_fmt: str = "%Y-%m-%d %H:%M:%S",
                 verbose: bool = True):
        super().__init__()

        # Params
        self.input_size_attr = input_size_attr
        self.output_size_attr = output_size_attr
        self.ops_attr = ops_attr
        self.inference_start_attr = inference_start_attr
        self.inference_end_attr = inference_end_attr
        self.io_size_fn_map = io_size_fn_map
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
            module (nn.Module): Input module.
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

    def _io_size_fn(self, module: nn.Module, input: Tuple[torch.Tensor],
                    output: Tuple[torch.Tensor]) -> None:
        """ Method used to obtain the input and output sizes of a
        ``nn.Module`` instance based on its class type and the function
        it is mapped to in ``io_size_fn_map``.

        Args:
            module (nn.Module): Input module.
            input (Tuple[torch.Tensor]): Input tensor(s).
            output (Tuple[torch.Tensor]): Output tensor(s).
        """
        # Obtain method to calculate io shapes
        if type(module) not in self.io_size_fn_map.keys():
            io_size_fn = self.io_size_fn_map["default"]

        else:
            io_size_fn = self.io_size_fn_map[type(module)]

        # Calculate io size
        input_size, output_size = io_size_fn(module, input, output)

        # Save input and output size in module attributes
        setattr(module, self.input_shape_attr, input_size)
        setattr(module, self.output_shape_attr, output_size)

    def _inference_time_start_fn(
                self,
                module: nn.Module,
                input: Tuple[torch.Tensor]
            ) -> None:
        """ Triggers a counter before performing the inference and save it's
        value to a module's attribute.

        .. note::
            Please note that this calculation may be affected by any other
            pre-forward hook attached to the module.

        Args:
            module (nn. Module): Input module.
            input (Tuple[torch.Tensor]): Input tensor(s) of the module's
                forward method.
        """
        setattr(module, self.inference_start_attr, perf_counter())

    def _inference_time_end_fn(
                self,
                module: nn.Module,
                input: Tuple[torch.Tensor],
                output: Tuple[torch.Tensor]
            ) -> None:
        """ Triggers a counter after the inference has been performed and save
        it's value to a module's attribute.

        .. note::
            Please note that this calculation may be affected by any other
            forward hook attached to the module.

        Args:
            module (nn.Module): Input module.
            input (Tuple[torch.Tensor]): Input tensor(s) of the module's
                forward method.
            output (Tuple[torch.Tensor]): Output tensor(s) of the module's
                forward method.
        """
        setattr(module, self.inference_end_attr, perf_counter())

    def _ops_fn(
                self,
                module: nn.Module,
                input: Tuple[torch.Tensor],
                output: Tuple[torch.Tensor]
            ) -> None:
        """ Triggers a method that estimates the number of operations computed
        by a module during the forward pass.

        Args:
            module (nn.Module): Input module.
            input (Tuple[torch.Tensor]): Input tensor(s) of the module's
                forward method.
            output (Tuple[torch.Tensor]): Output tensor(s) of the module's
                forward method.
        """
        # Obtain method to estimate ops
        if module.__class__ in self.ops_fn_map.keys():
            ops_fn = self.ops_fn_map[type(module)]

        else:
            ops_fn = self.ops_fn_map["default"]

        # Estimate ops
        ops_data = ops_fn(module, input, output)

        # Save ops in attribute
        setattr(module, self.ops_attr, ops_data)

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
        for idx, (n, m) in tqdm(enumerate(module.named_modules()),
                                desc="Counting parameters",
                                unit="params",
                                disable=not self.verbose,
                                leave=False):
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

    def count_params_df(self, *args, **kwargs) -> pd.DataFrame:
        """ Same as ``count_params`` but returns a ``DataFrame`` instead. """
        # Count params
        data = self.count_params(*args, **kwargs)

        # Assemble data frame
        df = pd.DataFrame()

        for k, v in data.items():
            row = {"module": k}
            row.update(v)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        return df

    def count_params_csv(self, file: str, *args, **kwargs) -> None:
        """ Same as ``count_params`` but saves a ``.csv`` file instead. """
        file = add_extension(file, ".csv")
        df = self.count_params_df(*args, **kwargs)
        df.to_csv(file, index=False)

    def count_params_html(self, file: str, *args, **kwargs) -> None:
        """ Same as ``count_params`` but saves a ``.html`` file instead. """
        file = add_extension(file, ".html")
        df = self.count_params_df(*args, **kwargs)

        with open(file, "w") as f:
            f.write(df.to_html())

    def count_params_latex(self, *args, index: bool = False, **kwargs) -> str:
        """ Same as ``count_params`` but returns a LaTeX output instead. """
        df = self.count_params_df(*args, **kwargs)
        return df.to_latex(index=index)

    @torch.no_grad()
    def estimate_inference_time(self):
        # TODO: Per layer inference time
        ...

    @torch.no_grad()
    def estimate_total_inference_time(
                self,
                module: nn.Module,
                input: Union[torch.Tensor, Tuple[torch.Tensor]],
                eval: bool = True,
                num_iters: int = 1000,
                drop_first: int = 100) -> dict:
        # Assertions
        if num_iters <= drop_first:
            raise ValueError(
                f"{num_iters=} should be greater than {drop_first=}"
            )

        if self.verbose:
            self._logger.log(
                "Estimating total inference time of "
                f"<b><magenta>{module.__class__.__name__}</magenta></b>"
            )

        # Setup
        if eval:
            if self.verbose:
                self._logger.log(
                    f"Setting module <b><magenta>{module.__class__.__name__}"
                    "</magenta></b> to <b><magenta>eval</magenta></b> mode"
                )

            was_training = True if module.training else False
            module.eval()

        # Store (start_time, end_time) tuples
        stopwatch = []

        # Compute inferences
        for _ in range(num_iters):
            start_time = perf_counter()
            module(input)
            end_time = perf_counter()
            stopwatch.append((start_time, end_time))

        # Tear down
        if eval and was_training:
            if self.verbose:
                self._logger.log(
                    f"Setting module <b><magenta>{module.__class__.__name__}"
                    "</magenta></b> to <b><magenta>train</magenta></b> mode"
                )

            module.train()

        # Collect stats
        data = {
            "__root__": {
                "type": module.__class__.__name__,
                "intraop_threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads()
            }
        }
        data["__root__"].update(get_hardware_specs())
        data["__root__"].update({
            "inference_time_ms":
            [(t[1] - t[0]) * 1000.0 for t in stopwatch[drop_first:]]
        })
        import pdb;pdb.set_trace()

        return data
