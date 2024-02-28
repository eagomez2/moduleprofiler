import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple, 
    Union
)
from time import perf_counter
from .utils import (
    make_list,
    dict_merge,
    add_extension,
    get_hardware_specs
)
from .logger import Logger
from .io_size import _DEFAULT_IO_SIZE_FN_MAP
from .ops import _DEFAULT_OPS_MAP


class ModuleProfiler:
    def __init__(
            self,
            input_size_attr: str = "__input_size__",
            output_size_attr: str = "__output_size__",
            ops_attr: str = "__ops__",
            inference_start_attr: str = "__inference_start__",
            inference_end_attr: str = "__inference_end__",
            io_size_fn_map: dict = _DEFAULT_IO_SIZE_FN_MAP,
            ops_fn_map: dict = _DEFAULT_OPS_MAP,
            ts_fmt: str = "%Y-%m-%d %H:%M:%S",
            verbose: bool = True
    ) -> None:
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

    def _setattr(
            self,
            module: nn.Module,
            attr: Union[str, list],
            value: Any = None
    ) -> None:
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
        """ Removes model attribute(s) if present.

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
        if dtype in [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64
        ]:
            bits = torch.iinfo(dtype).bits

        elif dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64
        ]:
            bits = torch.finfo(dtype).bits

        else:
            # NOTE: If you hit this point, please open a GitHub issue
            raise NotImplementedError

        return bits

    def _register_forward_hook(
            self,
            module: nn.Module,
            hook: Callable
    ) -> None:
        """ Registers a forward hook in a module and stores the corresponding
        handle to be deleted later.

        Args:
            module (nn.Module): Input module.
            hook (Callable): Hook to be registered.
        """
        self._hook_handles.append(module.register_forward_hook(hook))

    def _register_forward_pre_hook(
            self, module: nn.Module,
            hook: Callable
    ) -> None:
        """ Registers a forward pre hook in a module and stores the
        corresponding handle to be deleted later.

        Args:
            module (nn.Module): Input module.
            hook (Callable): Hook to be registered.
        """
        self._hook_handles.append(module.register_forward_pre_hook(hook))
    
    def _remove_registered_hooks(self) -> None:
        """ Removes all hooks registered by this object. """
        for hook_handle in self._hook_handles:
            hook_handle.remove()
        
        self._hook_handles = []

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
            if spec.keys() != ref_spec.keys():
                ref_spec_keys_repr =\
                    ", ".join([f"'{k}'" for k in ref_spec])
                spec_keys_repr =\
                    ", ".join([f"'{k}'" for k in spec])

                raise ValueError(
                    f"All specs to be merged should have the same keys. Found "
                    f"specs[0] and specs[{idx + 1}] have different keys: "
                    f"specs[0].keys()={ref_spec_keys_repr} and "
                    f"specs[{idx + 1}].keys()={spec_keys_repr}"
                )

        # Merge dicts
        merged_specs = ref_spec

        for spec in specs[1:]:
            for k in merged_specs:
                merged_specs[k] = dict_merge(merged_specs[k], spec[k])

        return merged_specs

    def _io_size_fn(
            self,
            module: nn.Module,
            input: Tuple[torch.Tensor],
            output: Tuple[torch.Tensor]
    ) -> None:
        """ Method used to obtain the input and output sizes of a
        ``nn.Module`` instance based on its class type and the function
        it is mapped to in ``io_size_fn_map``.

        Args:
            module (nn.Module): Input module.
            input (Tuple[torch.Tensor]): Input tensor(s).
            output (Tuple[torch.Tensor]): Output tensor(s).
        """
        # Obtain method to calculate io shapes
        if type(module) not in self.io_size_fn_map:
            io_size_fn = self.io_size_fn_map["default"]

        else:
            io_size_fn = self.io_size_fn_map[type(module)]

        # Calculate io size
        input_size, output_size = io_size_fn(module, input, output)

        # Save input and output size in module attributes
        setattr(module, self.input_size_attr, input_size)
        setattr(module, self.output_size_attr, output_size)

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
        if module.__class__ in self.ops_fn_map:
            ops_fn = self.ops_fn_map[type(module)]

        else:
            ops_fn = self.ops_fn_map["default"]

        # Estimate ops
        ops_data = ops_fn(module, input, output)

        # Save ops in attribute
        setattr(module, self.ops_attr, ops_data)

    def count_params(
            self,
            module: nn.Module,
            param_size: bool = True,
            param_dtype: bool = True,
            percent: bool = True
    ) -> dict:
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
        for idx, (n, m) in tqdm(
            enumerate(module.named_modules()),
            desc="Counting parameters",
            unit="params",
            disable=not self.verbose,
            leave=False
        ):
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

            for k in data:
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

        if self.verbose:
            self._logger.log(f"Results saved to <b>{file}</b>")

    def count_params_html(self, file: str, *args, **kwargs) -> None:
        """ Same as ``count_params`` but saves a ``.html`` file instead. """
        file = add_extension(file, ".html")
        df = self.count_params_df(*args, **kwargs)

        with open(file, "w") as f:
            f.write(df.to_html())
        
        if self.verbose:
            self._logger.log(f"Results saved to <b>{file}</b>")

    def count_params_latex(self, *args, index: bool = False, **kwargs) -> str:
        """ Same as ``count_params`` but returns a LaTeX output instead. """
        df = self.count_params_df(*args, **kwargs)
        return df.to_latex(index=index)

    @torch.no_grad()
    def estimate_inference_time(
        self,
        module: nn.Module,
        input: Union[torch.Tensor, Tuple[torch.Tensor]],
        eval: bool = True,
        num_iters: int = 1000,
        drop_first: int = 100
    ) -> dict:
        # Assertions
        if num_iters <= drop_first:
            raise ValueError(
                f"{num_iters=} should be greater than {drop_first=}"
            )
        
        if self.verbose:
            self._logger.log(
                "Estimating inference time of "
                f"<b><magenta>{module.__class__.__name__}</magenta></b>"
            )
        
        # Setup
        if eval:
            if self.verbose:
                self._logger.log(
                    f"Setting module <b><magenta>{module.__class__.__name__}"
                    "</magenta></b> to <b><magenta>eval</magenta></b> mode"
                )

            was_training = bool(module.training)
            module.eval()
        
        # Set attrs and hooks
        for m in module.modules():
            self._setattr(m, self.inference_start_attr)
            self._setattr(m, self.inference_end_attr)
            self._register_forward_pre_hook(m, self._inference_time_start_fn)
            self._register_forward_hook(m, self._inference_time_end_fn)

        # Measure
        # NOTE: Key '' is removed since it corresponds to the root module
        data = {
            "__root__": {
                "type": module.__class__.__name__,
                "inference_time_ms": []
            }
        }
        data.update(
            {n: {"type": m.__class__.__name__, "inference_time_ms": []}
             for n, m in module.named_modules()}
        )
        data.pop("")

        for _ in tqdm(
            range(num_iters),
            desc="Measuring inference time",
            unit="inferences",
            disable=not self.verbose,
            leave=False
        ):
            module(input)

            for n, m in module.named_modules():
                end = getattr(m, self.inference_end_attr)
                start = getattr(m, self.inference_start_attr)
                diff_ms = (end - start) * 1000.0
                k = "__root__" if n == "" else n
                data[k]["inference_time_ms"].append(diff_ms)

        # Tear down
        for m in module.modules():
            self._delattr(m, self.inference_start_attr)
            self._delattr(m, self.inference_end_attr)

        self._remove_registered_hooks()

        if eval and was_training:
            if self.verbose:
                self._logger.log(
                    f"Setting module <b><magenta>{module.__class__.__name__}"
                    "</magenta></b> to <b><magenta>train</magenta></b> mode"
                )

            module.train()
        
        for k in data:
            # Add time stats
            times = pd.Series(data[k]["inference_time_ms"])
            data[k].update(
                {
                    "inference_time_mean_ms": times.mean(),
                    "inference_time_max_ms": times.max(),
                    "inference_time_min_ms": times.min(),
                    "inference_time_std_ms": times.std(),
                    "inference_time_median_ms": times.median()
                }
            )

            # Add misc data
            data[k].update(
                {
                    "intraop_threads": torch.get_num_threads(),
                    "interop_threads": torch.get_num_interop_threads()
                }
            )
            data[k].update(get_hardware_specs())
        
        return data
    
    def estimate_inference_time_df(
            self,
            *args,
            aggr: bool = True,
            **kwargs
    ) -> pd.DataFrame:
        """ Same as ``estimate_inference_time`` but returns a ``DataFrame``
        instead. Additional argument ``aggr`` can be set to ``True`` if only
        aggregations should be kept.
        """
        # Estimate inference time
        data = self.estimate_inference_time(*args, **kwargs)

        # Assemble data frame
        df = pd.DataFrame()
        
        for k, v in data.items():
            row = {"module": k}
            row.update(v)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
        if aggr:
            df = df.drop("inference_time_ms", axis=1)

        else:
            for n in ("mean", "min", "max", "std", "median"):
                df = df.drop(f"inference_time_{n}_ms", axis=1)
            
            df = df.explode("inference_time_ms").reset_index(drop=True)
        
        return df
    
    def estimate_inference_time_csv(self, file: str, *args, **kwargs) -> None:
        """ Same as ``estimate_inference_time`` but saves a ``.csv`` file
        instead.
        """
        file = add_extension(file, ".csv")
        df = self.estimate_inference_time_df(*args, **kwargs)
        df.to_csv(file, index=False)

        if self.verbose:
            self._logger.log(f"Results saved to <b>{file}</b>")
    
    def estimate_inference_time_html(self, file: str, *args, **kwargs) -> None:
        """ Same as ``estimate_inference_time`` but saves a ``.html`` file
        instead.
        """
        file = add_extension(file, ".html")
        df = self.estimate_inference_time_df(*args, **kwargs)

        with open(file, "w") as f:
            f.write(df.to_html())

        if self.verbose:
            self._logger.log(f"Results saved to <b>{file}</b>")
        
    def estimate_inference_time_latex(
            self,
            *args,
            index: bool = False,
            **kwargs
    ) -> str:
        """ Same as ``estimate_inference_time`` but return a LaTeX output
        instead.
        """
        df = self.estimate_inference_time_df(*args, **kwargs)
        return df.to_latex(index=index)

    @torch.no_grad()
    def estimate_total_inference_time(
        self,
        module: nn.Module,
        input: Union[torch.Tensor, Tuple[torch.Tensor]],
        eval: bool = True,
        num_iters: int = 1000,
        drop_first: int = 100
    ) -> dict:
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

            was_training = bool(module.training)
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
        times = pd.Series(
            [(t[1] - t[0]) * 1000.0 for t in stopwatch[drop_first:]]
        )
        data = {
            "__root__": {
                "type": module.__class__.__name__,
                "intraop_threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads()
            }
        }
        data["__root__"].update(get_hardware_specs())
        data["__root__"].update({
            "inference_time_ms": times.tolist(),
            "inference_time_mean_ms": times.mean(),
            "inference_time_min_ms": times.min(),
            "inference_time_max_ms": times.max(),
            "inference_time_std_ms": times.std(),
            "inference_time_median_ms": times.median()
        })
        
        return data

    def estimate_total_inference_time_df(
            self,
            *args,
            aggr: bool = False,
            **kwargs
    ) -> pd.DataFrame:
        """ Same as ``estimate_total_inference_time`` but returns a
        ``DataFrame`` instead. Additional argument ``aggr`` can be set to
        ``True`` if only aggregations should be kept.
        """
        # Estimate inference total time
        data = self.estimate_total_inference_time(*args, **kwargs)

        # Assemble data frame
        df = pd.DataFrame()

        for k, v in data.items():
            row = {"module": k}
            row.update(v)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
        if aggr:
            df = df.drop("inference_time_ms", axis=1)

        else:
            for n in ("mean", "min", "max", "std", "median"):
                df = df.drop(f"inference_time_{n}_ms", axis=1)
            
            df = df.explode("inference_time_ms").reset_index(drop=True)
        
        return df

    def estimate_total_inference_time_csv(
            self,
            file: str,
            *args,
            **kwargs
    ) -> None:
        """ Same as ``estimate_total_inference_time`` but saves a ``.csv``
        file instead.
        """
        file = add_extension(file, ".csv")
        df = self.estimate_total_inference_time_df(*args, **kwargs)
        df.to_csv(file, index=False)

        if self.verbose:
            self._logger.log(f"Results saved to <b>{file}</b>")

    def estimate_total_inference_time_html(
            self,
            file: str,
            *args,
            **kwargs
    ) -> None:
        """ Same as ``estimate_total_inference_time`` but saves a ``.html``
        file instead.
        """ 
        file = add_extension(file, ".html")
        df = self.estimate_total_inference_time_df(*args, **kwargs)

        with open(file, "w") as f:
            f.write(df.to_html())
        
        if self.verbose:
            self._logger.log(f"Results saved to <b>{file}</b>")

    def estimate_total_inference_time_latex(
            self,
            *args,
            index: bool = False,
            **kwargs
    ) -> str:
        """ Same as ``estimate_total_inference_time`` but returns a LaTeX
        output instead.
        """
        df = self.estimate_inference_time_df(*args, **kwargs)
        return df.to_latex(index=index)

    @torch.no_grad()
    def trace_io_sizes(
            self,
            module: nn.Module,
            input: Union[
                torch.Tensor,
                Tuple[torch.Tensor],
                Dict[str, torch.Tensor]
            ],
            pred_fn: Optional[Callable] = None,
            eval: bool = False
    ) -> dict:
        if self.verbose:
            self._logger.log(
                "Tracing I/O shapes of <b><magenta>"
                f"{module.__class__.__name__}</magenta></b>"
            )

        # Set attrs and hooks
        for m in module.modules():
            self._setattr(m, self.input_size_attr)
            self._setattr(m, self.output_size_attr)
            self._register_forward_hook(m, self._io_size_fn)
        
        # Model setup
        if eval:
            if self.verbose:
                self._logger.log(
                    f"Setting module <b><magenta>{module.__class__.__name__}"
                    "</magenta></b> to <b><magenta>eval</magenta></b> mode"
                )

            was_training = bool(module.training)
            module.eval()
        
        # Trace I/O sizes
        if isinstance(input, dict):
            if pred_fn is not None:
                pred_fn(**input)
            
            else:
                module(**input)

        else:
            if pred_fn is not None:
                pred_fn(input)
            
            else:
                module(input)
            
        # Collect data
        data = {}

        for n, m in module.named_modules():
            k = "__root__" if n == "" else n
            data[k] = {
                "type": m.__class__.__name__,
                "input_size": getattr(m, self.input_size_attr),
                "output_size": getattr(m, self.output_size_attr)
            }
        
        # Tear down
        if eval and was_training:
            if self.verbose:
                self._logger.log(
                    f"Setting module <b><magenta>{module.__class__.__name__}"
                    "</magenta></b> to <b><magenta>train</magenta></b> mode"
                )

            module.train()
        
        for m in module.modules():
            self._delattr(m, self.input_size_attr)
            self._delattr(m, self.output_size_attr)
        
        self._remove_registered_hooks()

        return data

    def trace_io_sizes_df(self, *args, **kwargs) -> pd.DataFrame:
        """ Same as ``trace_io_sizes`` but returns a ``DataFrame`` instead. """
        # Trace I/O sizes
        data = self.trace_io_sizes(*args, **kwargs)

        # Assemble data frame
        df = pd.DataFrame()
    
        for k, v in data.items():
            row = {"module": k}
            row.update(v)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
        return df

    def trace_io_sizes_csv(self, file: str, *args, **kwargs) -> None:
        """ Same as ``trace_io_sizes`` but saves a ``.csv`` file instead. """
        file = add_extension(file, ".csv")
        df = self.trace_io_sizes_df(*args, **kwargs)
        df.to_csv(file, index=False)

        if self.verbose:
            self._logger.log(f"Results saved to <b>{file}</b>")

    def trace_io_sizes_html(self, file: str, *args, **kwargs) -> None:
        """ Same as ``trace_io_sizes`` but saves a ``.html`` file instead. """
        file = add_extension(file, ".html")
        df = self.trace_io_sizes_df(*args, **kwargs)
        
        with open(file, "w") as f:
            f.write(df.to_html())
        
        if self.verbose:
            self._logger.log(f"Results saved to <b>{file}</b>")
    
    def trace_io_sizes_latex(
            self,
            *args,
            index: bool = False,
            **kwargs
    ) -> str:
        """ Same as ``trace_io_sizes`` but returns a LaTeX output instead. """
        df = self.trace_io_sizes_df(*args, **kwargs)
        return df.to_latex(index=index)

    @torch.no_grad()
    def estimate_ops(
        self,
        module: nn.Module,
        input: Union[
            torch.Tensor,
            Tuple[torch.Tensor],
            Dict[str, torch.Tensor]
        ],
        pred_fn: Optional[Callable] = None,
        eval: bool = True
    ) -> dict:
        if self.verbose:
            self._logger.log(
                f"Estimating ops of <b><magenta>{module.__class__.__name__}"
                "</magenta></b>"
            )
        
        # Set attrs and hooks
        for m in module.modules():
            self._setattr(m, self.ops_attr)
            self._register_forward_hook(m, self._ops_fn)

        # Model setup
        if eval:
            if self.verbose:
                self._logger.log(
                    f"Setting module <b><magenta>{module.__class__.__name__}"
                    "</magenta></b> to <b><magenta>eval</magenta></b> mode"
                )

            was_training = bool(module.training)
            module.eval()

        # Estimate ops
        if isinstance(input, dict):
            if pred_fn is not None:
                pred_fn(**input)
            
            else:
                module(**input)
        
        else:
            if pred_fn is not None:
                pred_fn(input)
            
            else:
                module(input)

        # Collect data
        data = {}

        for n, m in module.named_modules():
            k = "__root__" if n == "" else n
            data[k] = {
                "type": m.__class__.__name__,
                "ops": getattr(m, self.ops_attr)
            }

        # Tear down
        if eval and was_training:
            if self.verbose:
                self._logger.log(
                    f"Setting module <b><magenta>{module.__class__.__name__}"
                    "</magenta></b> to <b><magenta>train</magenta></b> mode"
                )

            module.train()
        
        for m in module.modules():
            self._delattr(m, self.ops_attr)
        
        self._remove_registered_hooks()
    
        return data

    def estimate_ops_df(self, *args, **kwargs) -> pd.DataFrame:
        """ Same as ``estimate_ops`` but returns a ``DataFrame`` instead. """
        # Estimate ops
        data = self.estimate_ops(*args, **kwargs)

        # Assemble data frame
        df = pd.DataFrame()

        for k, v in data.items():
            row = {"module": k}
            row.update(v)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
        return df
    
    def estimate_ops_csv(self, file: str, *args, **kwargs) -> None:
        """ Same as ``estimate_ops`` but saves a ``.csv`` file instead. """
        file = add_extension(file, ".csv")
        df = self.estimate_ops_df(*args, **kwargs)
        df.to_csv(file, index=False)

        if self.verbose:
            self._logger.log(f"Results saved to <b>{file}</b>")

    def estimate_ops_html(self, file: str, *args, **kwargs) -> None:
        """ Same as ``estimate_ops`` but saves a ``.html`` file instead. """
        file = add_extension(file, ".html")
        df = self.estimate_ops_df(*args, **kwargs)

        with open(file, "w") as f:
            f.write(df.to_html())
        
        if self.verbose:
            self._logger.log(f"Results saved to <b>{file}</b>")

    def estimate_ops_latex(self, *args, index: bool = False, **kwargs) -> str:
        """ Same as ``estimate_ops`` but returns a LaTeX output instead. """
        df = self.estimate_ops_df(*args, **kwargs)
        return df.to_latex(index=index)
