# ConvTranspose1d (`torch.nn.ConvTranspose1d`)
A `torch.nn.ConvTranspose1d` modules applies a transposed convolution along a given dimension of a tensor. This operation can be seen as the gradient of a `torch.nn.Conv1d` convolution with respect to its input. Is it also known as <a href="https://en.wikipedia.org/wiki/Deconvolution" target="_blank">deconvolution</a>, however, this might be misleading because a deconvolution is the inverse of a convolution operation.

A `torch.nn.ConvTranspose1d` module expects an input of size $\left(N,C_{in}, L_{in}\right)$ or $\left(C_{in}, L_{in}\right)$ to produce an output of size $\left(N,C_{out}, L_{out}\right)$ or $\left(C_{out}, L_{out}\right)$. The relationship between layer parameters and $L_{out}$ is defined as

$$
\begin{equation}
    L_{out}=\left(L_{in}-1\right)\times \text{stride}-2\times\text{padding}+\text{dilation}\times\left(\text{kernel\_size}-1\right)+\text{output\_padding}+1
\end{equation}
$$

Where

* $N$ is the batch size.
* $C_{in}$ is the number of input channels.
* $C_{out}$ is the number of output channels.
* $L_{in}$ is the length of the input tensor (i.e. `x.size(-1)` assuming an input tensor `x`).
* $L_{out}$ is the length of the output tensor (i.e. `y.size(-1)` assuming an output tensor `y`).

The remaining parameters are assumed to be known by the reader and can be found in the <a href="https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html" target="_blank">torch.nn.ConvTranspose1d documentation</a>.


## Complexity

### Number of filters
In order to calculate the number of operations performed this module, it is necessary to understand the impact of the `groups` parameter on the overall complexity, and the number of filters $\psi$ a network instance will have based on this. According to <a href="https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html" target="_blank">the official `torch.nn.ConvTranspose1d` documentation</a>

> `groups` controls the connections between inputs and outputs. `in_channels` and `out_channels` must both be divisible by `groups`.

> For example:
> At `groups=1`, all inputs are convolved to all outputs.
> At `groups=2`, the operation becomes equivalent to having two conv(transpose1d) layers side by side, each seeing half the input channels and producing half the output channels, and both subsequently concatenated.
> 
> 
> At `groups=in_channels`, each input channel is convolved with its own set of filters
> (of size $\frac{\text{out\_channels}}{\text{in\_channels}}$ )
>

Based on this information, the number of filters $\psi$ can be computed as

$$
\begin{equation}
    \psi = \left(\frac{C_{in}}{\text{groups}}\right)\times\left(\frac{C_{out}}{\text{groups}}\right)\times{\text{groups}}=\frac{C_{in}\times C_{out}}{\text{groups}}
\end{equation}
$$

### Operations per filter
Now the [number of filters](#number-of-filters) $\psi$ are known, it is necessary to compute how many operations each filter performs. As shown in [Figure 1](#convtranspose1d-kernel-diagram), for each kernel position there will be $\text{kernel\_size}$ multiplications (i.e. each input element multiplied by the entire kernel). However, the additions pattern is more complicated compared to `torch.nn.Conv1d` because some kernel positions may overlap only partially. The possible outcomes become even more varied if the kernel has `dilation > 1` or `stride > 1`.

<figure markdown="span" id="convtranspose1d-kernel-diagram">
  ![conv1d-kernel-diagram](../figures/convtranspose1d-kernel-diagram-light.svg#only-light){ width="450" }
  ![conv1d-kernel-diagram](../figures/convtranspose1d-kernel-diagram-dark.svg#only-dark){ width="450" }
  <figcaption>Figure 1. Operations per kernel position to obtain the output tensor.</figcaption>
</figure>

For this case where obtaining a closed formula may be challenging, the approach to obtain the number of sums will be empirical. First, an input tensor $x$ will be filled with ones. Then, a `nn.ConvTranspose1d` will be instantiated, with `bias=False` and its kernel will also be initialized filled with ones. By doing this, it is possible to observe from [Figure 1](#convtranspose1d-kernel-diagram) that the output sequence will be $y_0=1$, $y_1=2$, $y_2=2$, and $y_3=1$. By subtracting $1$ to all values and adding the together, the result is the number of sums, $2$ in this case. Please find a code snippet below to illustrate this.

```py title="convtranspose1d_additions.py"
import torch

# Input tensor
x = torch.ones((1, 2))

# Module
convtranspose1d = torch.nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, bias=False)

# Fill weight with ones
torch.nn.init.ones_(convtranspose1d.weight)

# Compute number of additions
additions = convtranspose1d(x) - 1.0  # tensor([[0., 1., 1., 0.]])
additions = torch.sum(additions)  # tensor(2.)
```
Each element in $L_\text{out}$ is the result of $\text{kernel\_size}$ multiplications, and a number of additions that depends on possibly multiple overlapping kernel positions. The number of operations per filter $\lambda$ can be expressed as

$$
\begin{equation}
    \lambda=L_{out}\times\text{kernel\_size} + \text{additions\_per\_filter} 
\end{equation}
$$

Where $\text{additions\_per\_filter}$ corresponds to the result of the function to calculate the number of additions per filter.

!!! note
    Please note that the batch size $N$ will be ignored for now, but it will be included later on.

### Filter aggregation
Now that the [number of filters](#number-of-filters) and the number of [operations per filter](#operations-per-filter) are known, it is necessary compute the operations needed to aggregate each group of filters $\gamma$ to produce each output channel $C_{out}$. These operations correspond to simple element-wise additions and can be expressed as

$$
\begin{equation}
    \gamma=C_{out}\times L_\text{out}\times\left(\left(\frac{C_{in}}{\text{groups}}-1\right)+1\right)
\end{equation}
$$

Where the term $\left(\frac{C_{\text{in}}}{\text{groups}}-1\right)$ corresponds to the number of grouped connections between input and outputs channels $\frac{C_{\text{in}}}{\text{groups}}$, subtracted by $1$ because the operation is an addition. The $L_\text{out}$ factor accounts for the number of elements per filters, and $C_{\text{out}}$ expands this calculation to all output channels. Finally, the remaining $+1$ corresponds to the bias term $b$ that was not included so far, and that is added to each resulting output channel element. Note that this last term is only added if the module is instantiated using `bias=True`.

$$
\begin{equation}
\gamma=\begin{cases}
    C_{out}\times L_{out}\times\left(\frac{C_{in}}{\text{groups}}\right), & \text{if}\ \text{bias}=\text{True} \\
    C_{out}\times L_{out}\times\left(\frac{C_{in}}{\text{groups}}-1\right), & \text{if}\ \text{bias}=\text{False}
\end{cases}
\end{equation}
$$

!!! note
    Please note that the bias term $b$ was not included in  [Operations per filter](#operations-per-filter) and is added here instead. Even though according to <a href="https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html" target="_blank">PyTorch ``torch.nn.ConvTranspose1d`` documentation</a> $b$ has shape $\left(C_{out}\right)$, in practice this tensor is implicitly broadcasted following <a href="https://pytorch.org/docs/stable/notes/broadcasting.html" target="_blank">PyTorch broadcasting semantics</a> in such a way that each tensor value will be added with its corresponding channel bias.

### Total operations 
Now putting together all different factors that contribute to the total number of operations as well as including the batch size $N$

$$
\begin{equation}
    \text{ConvTranspose1d}_{ops}=N\times\left(\psi\times\lambda+\gamma\right)
\end{equation}
$$

Where

* $N$ is the batch size.
* $\psi$ is the [number of filters](#number-of-filters).
* $\lambda$ is the number of [operations per filter](#operations-per-filter).
* $\gamma$ is the number of [filter aggregation](#filter-aggregation) operations.

For the case of `bias=True` this can be expanded to

$$
\begin{equation}
    \small{\text{ConvTranspose1d}_{ops}=N\times\frac{C_{in}\times C_{out}}{\text{groups}}\times \left(L_{out}\times\left(\text{kernel\_size}+1\right)+\text{additions\_per\_filter}\right)}
\end{equation}
$$

For the case of `bias=False` $\gamma=C_{out}\times L_{out}\times\left(\frac{C_{in}}{\text{groups}}-1\right)$  and the whole expression can be simplified to

$$
\begin{equation}
    \small{\text{ConvTranspose1d}_{ops}=N\times\frac{C_{in}\times C_{out}}{\text{groups}}\times \left(L_{out}\times\left(\text{kernel\_size}+1\right)+\text{additions\_per\_filter}\right) - N\times C_{out}\times L_{out}}
\end{equation}
$$

## Summary
The number of operations performed by a `torch.nn.ConvTranspose1d` module can be estimated as

!!! success ""

    === "If `bias=True`"
        $\large{\text{ConvTranspose1d}_{ops}=N\times\frac{C_{in}\times C_{out}}{\text{groups}}\times \left(L_{out}\times\left(\text{kernel\_size}+1\right)+\text{additions\_per\_filter}\right)}$

    === "If `bias=False`"
        $\large{\text{ConvTranspose1d}_{ops}=N\times\frac{C_{in}\times C_{out}}{\text{groups}}\times \left(L_{out}\times\left(\text{kernel\_size}+1\right)+\text{additions\_per\_filter}\right) - N\times C_{out}\times L_{out}}$

Where

* $N$ is the batch size.
* $C_{\text{in}}$ is the number of input channels.
* $C_{\text{out}}$ is the number of output channels.
* $L_{\text{out}}$ is the length of the output tensor (i.e. `y.size(-1)` assuming an output tensor `y`).
* $\text{kernel\_size}$ is the length of the kernel.
* $\text{groups}$ is the number of groups.
* $\text{additions\_per\_filter}$ is the result of the function to calculate the number of addition operations per filter described in [Operations per filter](#operations-per-filter).