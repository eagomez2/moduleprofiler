"Module profiler"
__version__ = "0.0.2"

__all__ = [
    "get_default_ops_map",
    "get_default_io_size_map",
    "ModuleProfiler"
]

from .profiler import ModuleProfiler
from .ops import get_default_ops_map
from .io_size import get_default_io_size_map
