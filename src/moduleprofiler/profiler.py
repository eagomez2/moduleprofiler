from typing import Optional
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
