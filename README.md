# ModuleProfiler
Free open-source package to profile `torch.nn.Module` modules and obtain useful information to design a model that fits your needs and constraints at development time.

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

[<a href="https://eagomez2.github.io/moduleprofiler/" target="_blank">Online documentation</a> | <a href="https://eagomez2.github.io/moduleprofiler/tutorial/" target="_blank">Tutorial</a> ]

# Installation
`moduleprofiler` can be installed as any regular `python` module within your environment.

Install from PyPI:
```bash
python -m pip install moduleprofiler
```

Install from this repository:
```bash
python -m pip install git+https://github.com/eagomez2/moduleprofiler.git
```

# Documentation
You can access the <a href="https://eagomez2.github.io/moduleprofiler/" target="blank">online documentation</a>. There you will find a more in depth introduction to `moduleprofiler`, including tutorials, methods documentation and an extensive reference about the calculations utilized to estimate the operations of different supported `torch.nn.Module` modules.

You can also run the documentation locally by going to the root folder of the package and running:

```bash
mkdocs serve
```

Before running this, make sure that your python environment is enabled.

# Cite
If this package contributed to your work, please consider citing it:

```
@misc{moduleprofiler,
  author = {Esteban Gómez},
  title  = {moduleprofiler},
  year   = 2024,
  url    = {https://github.com/eagomez2/moduleprofiler}
}
```

# License
For further details about the license of this package, please see [LICENSE](LICENSE).