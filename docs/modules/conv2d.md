# Conv2d (`torch.nn.Conv2d`)

A `torch.nn.Conv2d` module applies the cross-correlation operation along a given pair of dimensions of a tensor. This may seem contradictory at first, because the module's name implies that the underlying operation should be convolution, yet both operations are similar.

!!! note
    Please note that the cross-correlation operation $\star$ is used instead of convolution $\ast$ even when the module name suggests the opposite. The main difference between these two operations is the kernel $g$ ordering, but the number of computations are equivalent. For this reason, we will use the term cross-correlation and convolution interchangeably hereafter.
    $$
        \left(f\ast g\right)[n]=\sum\limits_{k=0}^{K-1}f[n]\times g[n-k]\qquad \left(\text{convolution}\right)\\\~\\\
        \left(f\star g\right)[n]=\sum\limits_{k=0}^{K-1}f[n]\times g[n+k]\qquad \left(\text{cross-correlation}\right)
    $$

A `nn.Conv2d` module expects an input of size $\left(N,C_{\text{in}}, H_{\text{in}}, W_{\text{in}}\right)$ to produce an output of size $\left(N,C_{\text{out}}, H_{\text{out}}, W_{\text{out}}\right)$ performing the following operation

$$
\begin{equation}
    \text{out}\left(N_i, C_{\text{out}_j}\right) = \text{bias}\left(C_{\text{out}_j}\right) + \sum\limits_{k=0}^{C_{\text{in}}-1}\text{weight}\left(C_{\text{out}_j}, k\right) \star \text{input}\left(N_i, k\right)
\end{equation}
$$

Where

* $N$ is the batch size.
* $C_{\text{in}}$ is the number of input channels.
* $C_{\text{out}}$ is the number of output channels.
* $H_{\text{in}}$ is the height of the input tensor (i.e. `x.size(-2)` assuming an input tensor `x`)
* $W_{\text{in}}$ is the width of the input tensor (i.e. `x.size(-1)` assuming an input tensor `x`)
* $H_{\text{out}}$ is the height of the output tensor (i.e. `y.size(-2)` assuming an output tensor `y`)
* $W_{\text{out}}$ is the width of the output tensor (i.e. `y.size(-1)` assuming an output tensor `y`)
* $\star$ is the cross-correlation operator.

Additionally, $H_{\text{out}}$ (`y.size(-2)`) and $W_{\text{out}}$ (`y.size(-1)`)  will depend on $H_{\text{in}}$ (`x.size(-2)`), $W_{\text{in}}$ (`x.size(-1)`), `padding`, `dilation`, `kernel_size` and `stride` parameters. The relationship between these can be expressed as

$$
\begin{equation}
    H_\text{out}=\left[\frac{H_{\text{in}}+2\times\text{padding[0]} - \text{dilation[0]}\times\left(\text{kernel\_size[0]} - 1\right) - 1}{\text{stride[0]}}+1\right] \\
\end{equation}
$$

$$
\begin{equation}
    W_\text{out}=\left[\frac{W_{\text{in}}+2\times\text{padding[1]} - \text{dilation[1]}\times\left(\text{kernel\_size[1]} - 1\right) - 1}{\text{stride[1]}}+1\right]
\end{equation}
$$

Where indices $\text{[0]}$ and $\text{[1]}$ indicate first and second element of each tuple, respectively, since padding, dilation and kernel_size are now specified as tuples of two elements, compared to the nn.Conv1d case where they correspond to a single integer number.


## Complexity

### Number of filters

In order to calculate the number of operations performed this module, it is necessary to understand the impact of the `groups` parameter on the overall complexity, and the number of filters $\psi$ a network instance will have based on this. According to <a href="https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html" target="_blank">the official `torch.nn.Conv1d` documentation</a>

> `groups` controls the connections between inputs and outputs. `in_channels` and `out_channels` must both be divisible by `groups`.

> For example:
> At `groups=1`, all inputs are convolved to all outputs.
> At `groups=2`, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels and producing half the output channels, and both subsequently concatenated.
> 
> 
> At `groups=in_channels`, each input channel is convolved with its own set of filters
> (of size $\frac{\text{out\_channels}}{\text{in\_channels}}$ )
>

Based on this information, the number of filters $\psi$ can be computed as

$$
\begin{equation}
\psi = \left(\frac{C_{\text{in}}}{\text{groups}}\right)\times\left(\frac{C_{\text{out}}}{\text{groups}}\right)\times{\text{groups}}=\frac{C_{\text{in}}\times C_{\text{out}}}{\text{groups}}
\end{equation}
$$


### Operations per filter

Now the [number of filters](#number-of-filters) $\psi$ are known, it is necessary to compute how many operations each filter performs. As shown in [Figure 1](#conv2d-kernel-diagram), for each kernel position there will be $\text{kernel\_size}$ multiplications (i.e. each kernel element multiplied by a slice of the input tensor of the same size) and $\text{kernel\_size}-1$ additions to aggregate the result and obtain one element of the output.

<figure markdown="span" id="conv2d-kernel-diagram">
  ![conv1d-kernel-diagram](../figures/conv2d-kernel-diagram-light.svg#only-light){ width="750" }
  ![conv1d-kernel-diagram](../figures/conv2d-kernel-diagram-dark.svg#only-dark){ width="750" }
  <figcaption>Figure 1. Operations per kernel position to obtain the output tensor.</figcaption>
</figure>

Since each element in $\left(H_\text{out}, W_\text{out}\right)$ is the result of the operations carried out for a single kernel position, the number of operations per filter $\lambda$ can be expressed as

$$
\begin{equation}
    \lambda=\left(H_{\text{out}}\times W_{\text{out}}\right)\times\left(2\times\text{kernel\_size[0]}\times\text{kernel\_size[1]}-1\right)
\end{equation}
$$

Because the kernel is a 2-dimensional tensor of dimensions $\left(\text{kernel\_size[0]},\text{kernel\_size[1]}\right).$ 

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
    Please note that the bias term $b$ was not included in  [Operations per filter](#operations-per-filter) and is added here instead. Even though according to <a href="https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html" target="_blank">PyTorch ``torch.nn.Conv1d`` documentation</a> $b$ has shape $\left(C_\text{out}\right)$, in practice this tensor is implicitly broadcasted following <a href="https://pytorch.org/docs/stable/notes/broadcasting.html" target="_blank">PyTorch broadcasting semantics</a> in such a way that each tensor value will be added with its corresponding channel bias.


### Total operations

Now putting together all different factors that contribute to the total number of operations as well as including the batch size $N$

$$
\begin{equation}
    \text{Conv2d}_{ops}=N\times\left(\psi\times\lambda+\gamma\right)
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
    \scriptsize{\text{Conv2d}_{ops}=N\times\left(\left(\frac{C_{\text{in}}\times C_{\text{out}}}{\text{groups}}\right)\times\left(H_{\text{out}}\times W_\text{out}\times\left(2\times\text{kernel\_size[0]}\times\text{kernel\_size[1]}-1\right)\right)+C_{\text{out}}\times H_\text{out}\times W_\text{out}\times\left(\frac{C_{\text{in}}}{\text{groups}}\right)\right)}
\end{equation}
$$

Rearranging terms it can be simplified to

$$
\begin{equation}
\text{Conv2d}_{ops}=N\times\left(\frac{C_{\text{in}}\times C_{\text{out}}\times H_{\text{out}}\times W_\text{out}\times\left(2\times\text{kernel\_size[0]}\times\text{kernel\_size[1]}+1\right)}{\text{groups}}\right)
\end{equation}
$$

For the case of `bias=False` $\gamma=C_{\text{out}}\times H_\text{out}\times W_\text{out}\times\left(\frac{C_{\text{in}}}{\text{groups}}-1\right)$  and the whole expression can be simplified to

$$
\begin{equation}
\text{Conv2d}_{ops}=N\times\left(\frac{C_{\text{in}}\times C_{\text{out}}\times H_{\text{out}}\times W_\text{out}\times\left(2\times\text{kernel\_size[0]}\times\text{kernel\_size[1]}-\text{groups}\right)}{\text{groups}}\right)
\end{equation}
$$

## Summary

The number of operations performed by a `torch.nn.Conv2d` module can be estimated as

!!! success ""

    === "If `bias=True`"
        $\large{\text{Conv2d}_{ops}=N\times\left(\frac{C_{\text{in}}\times C_{\text{out}}\times H_{\text{out}}\times W_\text{out}\times\left(2\times\text{kernel\_size[0]}\times\text{kernel\_size[1]}+1\right)}{\text{groups}}\right)}$

    === "If `bias=False`"
        $\large{\text{Conv2d}_{ops}=N\times\left(\frac{C_{\text{in}}\times C_{\text{out}}\times H_{\text{out}}\times W_\text{out}\times\left(2\times\text{kernel\_size[0]}\times\text{kernel\_size[1]}-\text{groups}\right)}{\text{groups}}\right)}$

Where


* $N$ is the batch size.
* $C_{\text{in}}$ is the number of input channels.
* $C_{\text{out}}$ is the number of output channels.
* $H_{\text{in}}$ is the height of the input tensor (i.e. `x.size(-2)` assuming an input tensor `x`)
* $W_{\text{in}}$ is the width of the input tensor (i.e. `x.size(-1)` assuming an input tensor `x`)
* $\text{groups}$ is the number of groups.
* $\text{kernel\_size[0]}$ and $\text{kernel\_size[1]}$ are the first and second dimensions of the kernel tensor `weight`.