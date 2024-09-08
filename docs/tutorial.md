# Tutorial
This tutorial covers everything you need to know in order to use `moduleprofiler`. It begins with an introduction and explanation of the most basic methods, followed by a section on extending the calculations to your own custom `torch.nn.Module` modules. After completing this tutorial, you should be able to take advantage of this package and start using it in your own projects.

## Introduction
`moduleprofiler` is a free open-source package to profile `torch.nn.Module` modules and obtain useful information to design a model that fits your needs and constraints at development time. With `moduleprofiler` you can **calculate the number of parameters of your model**, **trace the input and output sizes of each layer**, **estimate the number of operations performed by each layer in the forward pass**, and **calculate the inference time**. The result of any profiling task can be obtained as a `dict` or `pandas.DataFrame` for further manipulation, can be exported to `.html` and `.csv` files, and can be used to generated a `LaTeX` table that you can add to your publication.

`moduleprofiler` works by iterating through all your module's `torch.nn.Module` instances. It uses a varied set of hooks that allows it to collect information abot your model. These hooks are temporarily added and then removed after performing all calculations. For this reason, `moduleprofiler` can work with any `torch.nn.Module` without needing any additional line of code, and without performing any permanent changes to your original model.

It comes equipped with all the necessary methods to deal with the most commonly used set of layers. However, you can also your own custom layers or modify the existing calculations to fit your particular needs.

## Basic usage

### Installation
`moduleprofiler` can be installed as any regular `python` module within your environment:

```bash
python -m pip install git+https://github.com/eagomez2/moduleprofiler.git
```

After doing this, all dependencies should be automatically resolved and you should be ready to start using it. In case of doubts about dependency version, you can inspect the `pyproject.toml` file in the root of this package.

### Basic profiling
All the basic functionality of `moduleprofiler` (and the most frequently required to profile a model) is encapsulated in a single class called `ModuleProfiler`.

```py
import torch
from moduleprofiler import ModuleProfiler

profiler = ModuleProfiler()
```

By default, a `ModuleProfiler` instance will be equipped with all you need, therefore no extra arguments are required, although you can inspect the available options in the [Documentation](documentation.md#moduleprofiler.profiler.ModuleProfiler) section.

After creating the required instance, you can already start profiling your module using one of the following methods:

* `profiler.count_params`
* `profiler.estimate_inference_time`
* `profiler.estimate_ops`
* `profiler.estimate_total_inference_time`
* `profiler.trace_io_sizes`

All these methods are available with different suffixes depending on the expected output format. For example, `profiler.count_params` will output a `dict` with all information about parameters in your module. On the other hand, `profiler.count_params_df` will output a `pandas.DataFrame` instead.


```py
import torch
from moduleprofiler import ModuleProfiler

# Profiler 
profiler = ModuleProfiler()

# Network
net = torch.nn.Linear(in_features=8, out_features=32)

# Number of parameters with dict output
params_dict = profiler.count_params(module=net)
"""
{'__root__':
    {
        'type': 'Linear',
        'trainable_params': 288,
        'nontrainable_params': 0,
        'trainable_params_dtype': torch.float32, 
        'trainable_params_size_bits': 9216,
        'trainable_params_percent': 1.0,
        'nontrainable_params_percent': 0.0
    }
}
"""

# Number of parameters with DataFrame output
params_df = profiler.count_params_df(module=net)
"""
     module    type  trainable_params  nontrainable_params trainable_params_dtype  trainable_params_size_bits  trainable_params_percent  nontrainable_params_percent
0  __root__  Linear               288                    0          torch.float32                        9216                       1.0                          0.0
"""
```

All results are consistent across formats, so it should be trivial to interpret them knowing the basics. You may see a `__root__` module in some of your results. This represents top-level `torch.nn.Module`.

Some methods such as `profiler.trace_io_sizes` or `profiler.estimate_ops` will require an example tensor when called, because the sizes of layer inputs and outputs will depend on this.

```py
# Example tensor
x = torch.rand((1, 8))

# Input and output sizes
io_sizes_df = profiler.trace_io_sizes_df(module=net, input=x)
"""
     module    type input_size output_size
0  __root__  Linear     (1, 8)     (1, 32)
"""

# Operations
ops_df = profiler.esimate_ops_df(module=net, input=x)
"""
     module    type  ops
0  __root__  Linear  512
"""
```

The complete code of this section is shown below:

```py title="basic_profiling.py"
import torch
from moduleprofiler import ModuleProfiler

# Profiler 
profiler = ModuleProfiler()

# Network
net = torch.nn.Linear(in_features=8, out_features=32)

# Number of parameters with dict output
params_dict = profiler.count_params(module=net)
"""
{'__root__':
    {
        'type': 'Linear',
        'trainable_params': 288,
        'nontrainable_params': 0,
        'trainable_params_dtype': torch.float32, 
        'trainable_params_size_bits': 9216,
        'trainable_params_percent': 1.0,
        'nontrainable_params_percent': 0.0
    }
}
"""

# Number of parameters with DataFrame output
params_df = profiler.count_params_df(module=net)
"""
     module    type  trainable_params  nontrainable_params trainable_params_dtype  trainable_params_size_bits  trainable_params_percent  nontrainable_params_percent
0  __root__  Linear               288                    0          torch.float32                        9216                       1.0                          0.0
"""

# Example tensor
x = torch.rand((1, 8))

# Input and output sizes
io_sizes_df = profiler.trace_io_sizes_df(module=net, input=x)
"""
     module    type input_size output_size
0  __root__  Linear     (1, 8)     (1, 32)
"""

# Operations
ops_df = profiler.estimate_ops_df(module=net, input=x)
"""
     module    type  ops
0  __root__  Linear  512
"""
```

### Inference time
Estimating inference time works as follows: First, your model is run multiple times and the time taken during each iteration is stored. After all runs are completed, some of the first iterations are dropped since these are typically slower than subsequent executions. This may be due to multiple reasons such as layer warm-up, benchmarking or caching mechanisms. By default, `1000` iterations are performed and the first `100` are dropped, but this number can configured by the user using the `num_iters` and `drop_first` parametrers. After this, only the relevant information is kept and all aggregations reported in the final output are based on these inference times.


```py
import torch
from moduleprofiler import ModuleProfiler

# Profiler 
profiler = ModuleProfiler()

# Network
net = torch.nn.Sequential(
    torch.nn.Linear(in_features=8, out_features=32),
    torch.nn.Sigmoid()
)

# Example tensor
x = torch.rand((1, 8))

# Compute inference time (aggregated)
aggr_inference_time_df = profiler.estimate_inference_time_df(module=net, input=x, num_iters=1000, drop_first=100)
"""
     module        type  inference_time_mean_ms  inference_time_max_ms  inference_time_min_ms  inference_time_std_ms  ...           host_name      os  os_release cpu_count total_memory          gpu
0  __root__  Sequential                0.008688               0.216167               0.007917               0.006605  ...  xxxxxxxxxxxx.local  Darwin      23.4.0        12     18432 MB  unavailable
1         0      Linear                0.003139               0.109833               0.002791               0.003388  ...  xxxxxxxxxxxx.local  Darwin      23.4.0        12     18432 MB  unavailable
2         1     Sigmoid                0.001511               0.028208               0.001333               0.000852  ...  xxxxxxxxxxxx.local  Darwin      23.4.0        12     18432 MB  unavailable
"""

# Compute inference time
inference_time_df = profiler.estimate_inference_time_df(module=net, input=x, num_iters=1000, drop_first=100, aggr=False)
"""
        module        type inference_time_ms  intraop_threads  interop_threads           host_name      os os_release  cpu_count total_memory          gpu
0     __root__  Sequential          0.040958                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
1     __root__  Sequential             0.011                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
2     __root__  Sequential          0.010333                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
3     __root__  Sequential          0.009791                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
4     __root__  Sequential          0.009916                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
...        ...         ...               ...              ...              ...                 ...     ...        ...        ...          ...          ...
2995         1     Sigmoid          0.001584                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
2996         1     Sigmoid          0.001584                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
2997         1     Sigmoid          0.001541                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
2998         1     Sigmoid          0.001583                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
2999         1     Sigmoid          0.001542                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
"""
```

Please note that there is some extra information besides the aggregated statistics of the inference time of your model. Specifications such as `host_name`, `os_release` (operating system version) and `cpu_count` (total number of cores) is added to the results.

If instead of aggregated statistics you want to obtain the time taken by each module during each inference, you can set `aggr=False` in `profiler.estimate_inference_time_df`. In this case, the first `drop_first` will also be included in the results and you will obtain a `DataFrame` with `num_iters * num_layers` rows.


!!! warning
    Please note that the inference of time of your model is hardware-dependent. This means that you may get significantly different values if you run the same model in different devices. Additionally, you may get significantly different numbers depending on the amount of configured intra-op and inter-op threads. To configure these you can use `torch.get_num_threads()`, `torch.set_num_threads()`, `torch.get_num_interop_threads()` and `torch.set_num_interop_threads()`. For further details about these methods, please refer to PyTorch documentation.

In addition to `profiler.estimate_inference_time` or `profiler.estimate_inference_time_df` there is a `profiler.estimate_total_inference_time` and `profiler.estimate_total_inference_time_df` method. This method represents a shortcut to obtaining statistics for the whole model, rather than computing them layer by layer. The method is recommended when you only need to compute the inference time for the whole model rather than layer by layer. Similarly to `profiler.esimate_inference_time`, you can compute individual inferences or aggregated metrics using the `aggr` parameter.

```py
# Compute inference time (aggregated)
aggr_total_inference_time_df = profiler.estimate_total_inference_time_df(module=net, input=x, num_iters=1000, drop_first=100, aggr=True)
"""
     module        type  intraop_threads  interop_threads           host_name  ... inference_time_mean_ms inference_time_min_ms  inference_time_max_ms inference_time_std_ms inference_time_median_ms
0  __root__  Sequential                6               12  xxxxxxxxxxxx.local  ...               0.004034              0.003833               0.011959              0.000361                    0.004
"""

# Compute inference time
total_inference_time_df = profiler.estimate_total_inference_time_df(module=net, input=x, num_iters=1000, drop_first=100)
"""
       module        type  intraop_threads  interop_threads           host_name      os os_release  cpu_count total_memory          gpu inference_time_ms
0    __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable             0.004
1    __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable          0.004041
2    __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable             0.004
3    __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable             0.004
4    __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable          0.004042
..        ...         ...              ...              ...               ...     ...        ...        ...          ...          ...               ...
895  __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable          0.005375
896  __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable          0.005375
897  __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable          0.004958
898  __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable          0.005125
899  __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable           0.00525
"""
```

The complete code of this section is shown below:

```py title="inference_time.py"
import torch
from moduleprofiler import ModuleProfiler

# Profiler 
profiler = ModuleProfiler()

# Network
net = torch.nn.Sequential(
    torch.nn.Linear(in_features=8, out_features=32),
    torch.nn.Sigmoid()
)

# Example tensor
x = torch.rand((1, 8))

# Compute inference time (aggregated)
aggr_inference_time_df = profiler.estimate_inference_time_df(module=net, input=x, num_iters=1000, drop_first=100)
"""
     module        type  inference_time_mean_ms  inference_time_max_ms  inference_time_min_ms  inference_time_std_ms  ...           host_name      os  os_release cpu_count total_memory          gpu
0  __root__  Sequential                0.008688               0.216167               0.007917               0.006605  ...  xxxxxxxxxxxx.local  Darwin      23.4.0        12     18432 MB  unavailable
1         0      Linear                0.003139               0.109833               0.002791               0.003388  ...  xxxxxxxxxxxx.local  Darwin      23.4.0        12     18432 MB  unavailable
2         1     Sigmoid                0.001511               0.028208               0.001333               0.000852  ...  xxxxxxxxxxxx.local  Darwin      23.4.0        12     18432 MB  unavailable
"""

# Compute inference time
inference_time_df = profiler.estimate_inference_time_df(module=net, input=x, num_iters=1000, drop_first=100, aggr=False)
"""
        module        type inference_time_ms  intraop_threads  interop_threads           host_name      os os_release  cpu_count total_memory          gpu
0     __root__  Sequential          0.040958                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
1     __root__  Sequential             0.011                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
2     __root__  Sequential          0.010333                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
3     __root__  Sequential          0.009791                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
4     __root__  Sequential          0.009916                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
...        ...         ...               ...              ...              ...                 ...     ...        ...        ...          ...          ...
2995         1     Sigmoid          0.001584                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
2996         1     Sigmoid          0.001584                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
2997         1     Sigmoid          0.001541                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
2998         1     Sigmoid          0.001583                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
2999         1     Sigmoid          0.001542                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable
"""

# Compute inference time (aggregated)
aggr_total_inference_time_df = profiler.estimate_total_inference_time_df(module=net, input=x, num_iters=1000, drop_first=100, aggr=True)
"""
     module        type  intraop_threads  interop_threads           host_name  ... inference_time_mean_ms inference_time_min_ms  inference_time_max_ms inference_time_std_ms inference_time_median_ms
0  __root__  Sequential                6               12  xxxxxxxxxxxx.local  ...               0.004034              0.003833               0.011959              0.000361                    0.004
"""

# Compute inference time
total_inference_time_df = profiler.estimate_total_inference_time_df(module=net, input=x, num_iters=1000, drop_first=100)
"""
       module        type  intraop_threads  interop_threads           host_name      os os_release  cpu_count total_memory          gpu inference_time_ms
0    __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable             0.004
1    __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable          0.004041
2    __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable             0.004
3    __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable             0.004
4    __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable          0.004042
..        ...         ...              ...              ...               ...     ...        ...        ...          ...          ...               ...
895  __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable          0.005375
896  __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable          0.005375
897  __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable          0.004958
898  __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable          0.005125
899  __root__  Sequential                6               12  xxxxxxxxxxxx.local  Darwin     23.4.0         12     18432 MB  unavailable           0.00525
"""
```

### Running documentation
Besides the available online documentation, you can run your own copy locally. This might be helpful for quicker inspection or while you are working offline. After installing all dependencies, simply go to the root of this package and run

```bash
mkdocs serve
```

If everything run correctly, you should see a message in your terminal containing the URL where the documentation will be served.

```
INFO    -  Building documentation...
INFO    -  Cleaning site directory
INFO    -  Documentation built in 0.39 seconds
INFO    -  [17:14:19] Watching paths for changes: 'docs', 'mkdocs.yml'
INFO    -  [17:14:19] Serving on http://127.0.0.1:8000/
```

Finally, just open the URL shown in the screen in your browser of choice.

### Running unit tests
`moduleprofiler` relies on `pytest` for running unit tests. All available tests are in the `tests` folder. To run all tests from the terminal, simpy enable your python environment, go to the package's root and run

```bash
pytest -v
```

This command will automatically detect and collect all test and run them. The results will be displayed in your terminal. 

## Advanced usage
`moduleprofiler` comes batteries-included. It already has supports for all the most common `torch.nn.Module` types. However, there are cases where it is necessary to add custom calculations, such as when you create a custom `torch.nn.Module` and want to include it in the estimations `moduleprofiler` can provide. The next sections explain how to add your custom module to both ops estimation and io size tracing. 

### Extending ops estimation
`moduleprofiler` uses a `dict` that maps `torch.nn.Module` types to a function that is called when the number of operations are computed. The function signature is always as follows:

```py
def ops_fn(module: torch.nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> int:
     # Your function body
     ...
```

!!! note
     Please note that this is the same signature as PyTorch forward hooks. If the input to the `forward` method of your module is a single `torch.Tensor`, the function will receive a `tuple` of one item.

To include your custom module (or overwrite existing calculations), you simply need to get the default `dict`, add your custom mapping, and pass it to the `ModuleProfiler` instance.

```py title="custom_module_ops.py"
import torch
from typing import Tuple
from moduleprofiler import (
     ModuleProfiler,
     get_default_ops_map
)


class PlusOne(torch.nn.Module):
    """A custom module that adds one to all items in the input tensor."""
    def __init__(self) -> None:
        super().__init__()
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


def plusone_ops_fn(
        module: PlusOne,
        input: Tuple[torch.Tensor],
        output: torch.Tensor
) -> int:
    # One operation per item
    return input[0].numel()

# Get default ops map
custom_ops_fn_map = get_default_ops_map()

# Register new function
custom_ops_fn_map[PlusOne] = plusone_ops_fn

# Create profiler with updated ops map
profiler = ModuleProfiler(ops_fn_map=custom_ops_fn_map)

# Create network with custom module
net = PlusOne()

# Create example tensor
x = torch.rand((1, 8))

# Compute ops
ops_df = profiler.estimate_ops_df(module=net, input=x)
"""
     module     type  ops
0  __root__  PlusOne    8
"""
```


### Extending io size tracing
