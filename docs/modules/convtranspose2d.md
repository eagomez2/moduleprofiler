# ConvTranspose2d (`torch.nn.ConvTranspose2d`)
A `torch.nn.ConvTranspose2d` modules applies a transposed convolution along a given pair of dimensions of a tensor. This operation can be seen as the gradient of a `torch.nn.Conv2d` convolution with respect to its input. Is it also known as <a href="https://en.wikipedia.org/wiki/Deconvolution" target="_blank">deconvolution</a>, however, this might be misleading because a deconvolution is the inverse of a convolution operation.

A `torch.nn.ConvTranspose2d` module expects an input of size $\left(N,C_{in}, H_{in}, W_{in}\right)$ or $\left(C_{in}, H_{in}, W_{in}\right)$ to produce an output of size $\left(N,C_{out}, H_{out}, W_{out}\right)$ or $\left(C_{out}, H_{out}, W_{out}\right)$. The relationship between layer parameters, $H_{out}$ and $W_{out}$ is defined as

$$
\begin{equation}
    \small{
        H_{out}=\left(H_{in}-1\right)\times \text{stride[0]}-2\times\text{padding[0]}+\text{dilation[0]}\times\left(\text{kernel\_size[0]}-1\right)+\text{output\_padding[0]}+1
    }
\end{equation}
$$

$$
\begin{equation}
    \small{
        W_{out}=\left(W_{in}-1\right)\times \text{stride[1]}-2\times\text{padding[1]}+\text{dilation[1]}\times\left(\text{kernel\_size[1]}-1\right)+\text{output\_padding[1]}+1
    }
\end{equation}
$$

Where

* $N$ is the batch size.
* $C_{in}$ is the number of input channels.
* $C_{out}$ is the number of output channels.
* $H_{in}$ is the height of the input tensor (i.e. `x.size(-2)` assuming an input tensor `x`)
* $W_{in}$ is the width of the input tensor (i.e. `x.size(-1)` assuming an input tensor `x`)
* $H_{out}$ is the height of the output tensor (i.e. `y.size(-2)` assuming an output tensor `y`)
* $W_{out}$ is the width of the output tensor (i.e. `y.size(-1)` assuming an output tensor `y`)

The remaining parameters are assumed to be known by the reader and can be found in the <a href="https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html" target="_blank">torch.nn.ConvTranspose2d documentation</a>.


## Complexity

### Number of filters
In order to calculate the number of operations performed this module, it is necessary to understand the impact of the `groups` parameter on the overall complexity, and the number of filters $\psi$ a network instance will have based on this. According to <a href="https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html" target="_blank">the official `torch.nn.ConvTranspose2d` documentation</a>

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
Now the [number of filters](#number-of-filters) $\psi$ are known, it is necessary to compute how many operations each filter performs. For each kernel position there will be $\text{kernel\_size[0]}\times\text{kernel\_size[1]}$ multiplications (i.e. each input element multiplied by the entire kernel). However, the additions pattern is more complicated compared to `torch.nn.Conv2d` because some kernel positions may overlap only partially. The possible outcomes become even more varied if the kernel has `dilation > 1` or `stride > 1`. Please see <a href="https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md", target="_blank">these animations</a> to visually understand how these patterns occur. It is the 2-dimensional generalization of the patterns already shown in [`torch.nn.ConvTranspose1d` complexity calculations](convtranspose1d.md#operations-per-filter).

For this case where obtaining a closed formula may be challenging, the approach to obtain the number of sums will be empirical. First, an input tensor $x$ will be filled with ones. Then, a `nn.ConvTranspose2d` will be instantiated, with `bias=False` and its kernel will also be initialized filled with ones. By doing this, it is possible to observe that similarly to the case of [`nn.ConvTranspose1d`](convtranspose1d.md#operations-per-filter), we can obtain the additions pattern by subtracting $1$ to all values, and adding the together. Please find a code snippet below to illustrate this.

```py title="convtranspose2d_additions.py"
import torch

# Input tensor
x = torch.ones((1, 2, 2))

# Module
convtranspose2d = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)

# Fill weight with ones
torch.nn.init.ones_(convtranspose2d.weight)

# Compute number of additions
additions = convtranspose2d(x) - 1.0  # tensor([[[0., 1., 0.], [1., 3., 1.], [0., 1., 0.]]])
additions = torch.sum(additions)  # tensor(7.)
```

Each element in $H_{out}\times W_{out}$ is the result of $\text{kernel\_size[0]}\times\text{kernel\_size[1]}$ multiplications, and a number of additions that depends on possibly multiple overlapping kernel positions. The number of operations per filter $\lambda$ can be expressed as

$$
\begin{equation}
    \lambda=H_{out}\times W_{out}\times\text{kernel\_size[0]}\times\text{kernel\_size[1]} + \text{additions\_per\_filter} 
\end{equation}
$$

Where $\text{additions\_per\_filter}$ corresponds to the result of the function to calculate the number of additions per filter.

!!! note
    Please note that the batch size $N$ will be ignored for now, but it will be included later on.

### Filter aggregation
Now that the [number of filters](#number-of-filters) and the number of [operations per filter](#operations-per-filter) are known, it is necessary compute the operations needed to aggregate each group of filters $\gamma$ to produce each output channel $C_\text{out}$. These operations correspond to simple element-wise additions and can be expressed as

$$
\begin{equation}
    \gamma=C_{\text{out}}\times H_\text{out}\times W_\text{out}\times\left(\left(\frac{C_{\text{in}}}{\text{groups}}-1\right)+1\right)
\end{equation}
$$

Where the term $\left(\frac{C_{\text{in}}}{\text{groups}}-1\right)$ corresponds to the number of grouped connections between input and outputs channels $\frac{C_{\text{in}}}{\text{groups}}$, subtracted by $1$ because the operation is an addition. The $H_\text{out}\times W_\text{out}$ factor accounts for the number of elements per filters, and $C_{\text{out}}$ expands this calculation to all output channels. Finally, the remaining $+1$ corresponds to the bias term $b$ that was not included so far, and that is added to each resulting output channel element. Note that this last term is only added if the module is instantiated using `bias=True`.

$$
\begin{equation}
\gamma=\begin{cases}
    C_{\text{out}}\times H_\text{out}\times W_\text{out}\times\left(\frac{C_{\text{in}}}{\text{groups}}\right), & \text{if}\ \text{bias}=\text{True} \\
    C_{\text{out}}\times H_\text{out}\times W_\text{out}\times\left(\frac{C_{\text{in}}}{\text{groups}}-1\right), & \text{if}\ \text{bias}=\text{False}
\end{cases}
\end{equation}
$$

!!! note
    Please note that the bias term $b$ was not included in  [Operations per filter](#operations-per-filter) and is added here instead. Even though according to <a href="https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html" target="_blank">PyTorch `torch.nn.ConvTranspose2d` documentation</a> $b$ has shape $\left(C_\text{out}\right)$, in practice this tensor is implicitly broadcasted following <a href="https://pytorch.org/docs/stable/notes/broadcasting.html" target="_blank">PyTorch broadcasting semantics</a> in such a way that each tensor value will be added with its corresponding channel bias.


### Total operations
Now putting together all different factors that contribute to the total number of operations as well as including the batch size $N$

$$
\begin{equation}
    \text{ConvTranspose2d}_{ops}=N\times\left(\psi\times\lambda+\gamma\right)
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
    \scriptsize{
        \text{ConvTranspose2d}_{ops} = N \times \frac{C_{in} \times C_{out}}{\text{groups}} \times \left( H_{out} \times W_{out} \times \left( \text{kernel\_size}[0] \times \text{kernel\_size}[1] + 1 \right) + \text{additions\_per\_filter} \right)
    }
\end{equation}
$$

For the case of `bias=False` $\gamma=C_{out}\times H_{out}\times W_{out}\times\left(\frac{C_{in}}{\text{groups}}-1\right)$  and the whole expression can be simplified to

$$
\begin{equation}
    \scriptsize{
        \text{ConvTranspose2d}_{ops} = N \times \frac{C_{in} \times C_{out}}{\text{groups}} \times \left( H_{out} \times W_{out} \times \left( \text{kernel\_size}[0] \times \text{kernel\_size}[1] + 1 \right) + \text{additions\_per\_filter} \right) -  N\times C_{out}\times H_{out}\times W_{out}
    }
\end{equation}
$$

## Summary
The number of operations performed by a `torch.nn.ConvTranspose2d` module can be estimated as

!!! success ""

    === "If `bias=True`"
        $\small{\text{ConvTranspose2d}_{ops} = N \times \frac{C_{in} \times C_{out}}{\text{groups}} \times \left( H_{out} \times W_{out} \times \left( \text{kernel\_size}[0] \times \text{kernel\_size}[1] + 1 \right) + \text{additions\_per\_filter} \right)}$

    === "If `bias=False`"
        $\small{\text{ConvTranspose2d}_{ops} = N \times \frac{C_{in} \times C_{out}}{\text{groups}} \times \left( H_{out} \times W_{out} \times \left( \text{kernel\_size}[0] \times \text{kernel\_size}[1] + 1 \right) + \text{additions\_per\_filter} \right) -  N\times C_{out}\times H_{out}\times W_{out}}$

Where

* $N$ is the batch size.
* $C_{in}$ is the number of input channels.
* $C_{out}$ is the number of output channels.
* $H_{out}$ is the height of the output tensor (i.e. `y.size(-2)` assuming an output tensor `x`)
* $W_{out}$ is the width of the output tensor (i.e. `y.size(-1)` assuming an output tensor `x`)
* $\text{kernel\_size[0]}$ and $\text{kernel\_size[1]}$ are the first and second dimensions of the `kernel_size` tuple.
* $\text{groups}$ is the number of groups.
* $\text{additions\_per\_filter}$ is the result of the function to calculate the number of addition operations per filter described in [Operations per filter](#operations-per-filter).