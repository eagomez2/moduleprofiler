# About
`moduleprofiler` is a free open-source package to profile `torch.nn.Module` modules and obtain useful information to design a model that fits your needs and constraints at development time.

With `moduleprofiler` you can:

* Calculate the number of parameters of your model.
* Trace the input and output sizes of each component of your model.
* Estimate the number of operations your model performs in a forward pass.
* Calculate per module and total inference time.

All results can be obtained in one of the following formats:

* `dict` (default output format)
* `pandas.DataFrame` (to perform further calculations or filtering in your code)
* `html` (to export as webpage)
* `LaTeX` (to include in your publications)

## Installation
`moduleprofiler` can be installed as any regular `python` module within your environment:

```bash
python -m pip install git+https://github.com/eagomez2/moduleprofiler.git
```

## Quickstart
Using `moduleprofiler` is simple. First, you need to create a `ModuleProfiler`
object, and then you can run any method over your existing `torch.nn.Module` model.

```py title="basic_moduleprofiler_example.py"
import torch
from moduleprofiler import ModuleProfiler

# Input tensor
x = torch.rand((1, 8))

# Network
net = torch.nn.Linear(in_features=8, out_features=32)

# Profiler
profiler = ModuleProfiler()

# Calculate number of parameters in the model
net_params = profiler.count_params(module=net)
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

# Compute sizes of input and output tensors of each layer
net_io = profiler.trace_io_sizes(module=net, input=x)
"""
{'__root__':
  {'type': 'Linear', 'input_size': (1, 8), 'output_size': (1, 32)}
}
"""

# Estimate operations performed by each layer
net_ops = profiler.estimate_ops(module=net, input=x)
"""
{'__root__': {'type': 'Linear', 'ops': 512}}
"""
```

Methods for generating the results in different formats such as `DataFrame`, `.csv` or `.html` have the same names, except for a suffixed ending. For example, [`profiler.trace_io_sizes`](documentation.md/#moduleprofiler.profiler.ModuleProfiler.trace_io_sizes) becomes [`profiler.trace_io_sizes_df`](documentation.md/#moduleprofiler.profiler.ModuleProfiler.trace_io_sizes_df) for a `DataFrame` output, or [`profiler.trace_io_sizes_csv`](documentation.md/#moduleprofiler.profiler.ModuleProfiler.trace_io_sizes_csv) to save results to a `.csv` file. Methods without a suffix will generate a `dict` output.

* For more in-depth tutorials please see the [Tutorial](tutorial.md) section.
* For further information about specifics methods, please see the [Documentation](documentation.md) section.
* For further information about how the complexity of different layers is
estimated, please see the [Reference](reference.md) section.

If you have any feedback or suggestions, feel free to open an issue in GitHub or reach out to the author. If this package contributed to your work, please consider citing it:

```
@misc{moduleprofiler,
  author = {Esteban Gómez},
  title  = {moduleprofiler},
  year   = 2024,
  url    = {https://github.com/eagomez2/moduleprofiler}
}
```

## Supported modules
By default, all methods support all modules as long as these are instances of `torch.nn.Module`. The only functionality where some modules might be missing is in the [`estimate_ops`](documentation.md#moduleprofiler.profiler.ModuleProfiler.estimate_ops) method that used to estimate the operations performed by a certain module. Modules that are not supported will result in `None` as returned value. However, it is possible to add both missing and custom modules as described in [Tutorial](tutorial.md#extending-ops-estimation).


| Module                       | Supported (ops)                    | Version |
| ---------------------------- | ---------------------------------- | ------- |
| `torch.nn.Identity`          | :material-check:                   | 0.0.1   |
| `torch.nn.Linear`            | :material-check:                   | 0.0.1   |
| `torch.nn.Conv1d`            | :material-check:                   | 0.0.1   |
| `torch.nn.Conv2d`            | :material-check:                   | 0.0.1   |
| `torch.nn.ConvTranspose1d`   | :material-check:                   | 0.0.1   |
| `torch.nn.ConvTranspose2d`   | :material-check:                   | 0.0.1   |
| `torch.nn.GRUCell`           | :material-check:                   | 0.0.1   |
| `torch.nn.GRU`               | :material-check:                   | 0.0.1   |
| `torch.nn.LSTMCell`          | :material-check:                   | 0.0.1   |
| `torch.nn.LSTM`              | :material-check:                   | 0.0.1   |
| `torch.nn.MultiheadAttention`| :material-close:                   |         |
| `torch.nn.ReLU`              | :material-check:                   | 0.0.1   |
| `torch.nn.LeakyReLU`         | :material-check:                   | 0.0.1   |
| `torch.nn.ELU`               | :material-check:                   | 0.0.1   |
| `torch.nn.PReLU`             | :material-check:                   | 0.0.1   |
| `torch.nn.Sigmoid`           | :material-check:                   | 0.0.1   |
| `torch.nn.Softmax`           | :material-check:                   | 0.0.1   |
| `torch.nn.Softplus`          | :material-check:                   | 0.0.1   |
| `torch.nn.Tanh`              | :material-check:                   | 0.0.1   |
| `torch.nn.AdaptiveMaxPool1d` | :material-check:                   | 0.0.1   |
| `torch.nn.AdaptiveMaxPool2d` | :material-check:                   | 0.0.1   |
| `torch.nn.MaxPool1d`         | :material-check:                   | 0.0.1   |
| `torch.nn.MaxPool2d`         | :material-check:                   | 0.0.1   |