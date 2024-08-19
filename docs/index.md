# About

`moduleprofiler` is a free open-source package to profile `torch.nn.Module` modules and obtain useful information to design a model that fits your needs and constraints at development time.

With `moduleprofiler` you can:

- Calculate the number of parameters of your model.
- Trace the input and output sizes of each component of your model.
- Estimate the number of operations your model performs in a forward pass.
- Calculate per module and total inference time.

All results can be obtained in one of the following formats:

- `dict` (default output format)
- `pandas.DataFrame` (to perform further calculations or filtering in your code)
- `html` (to export as webpage)
- `LaTeX` (to include in your publications)

## Installation

`moduleprofiler` can be installed as any regular `python` module within your environment:

```bash
python -m pip install git+https://github.com/eagomez2/moduleprofiler.git
```

## Quickstart
...
