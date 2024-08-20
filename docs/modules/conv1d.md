# Conv1d (`torch.nn.Conv1d`)
A `torch.nn.Conv1d` module applies the cross-correlation operation along a given dimension of a tensor. This may seem contradictory at first, because the module's name implies that the underlying operation should be convolution, yet both operations are similar.

!!! note
    Please note that the cross-correlation operation $\star$ is used instead of convolution $\ast$ even when the module name suggests the opposite. The main difference between these two operations is the kernel $g$ ordering, but the number of computations are equivalent. For this reason, we will use the term cross-correlation and convolution interchangeably hereafter.
    $$
        \left(f\ast g\right)[n]=\sum\limits_{k=0}^{K-1}f[n]\times g[n-k]\qquad \left(\text{convolution}\right)\\\~\\\
        \left(f\star g\right)[n]=\sum\limits_{k=0}^{K-1}f[n]\times g[n+k]\qquad \left(\text{cross-correlation}\right)
    $$

A `torch.nn.Conv1d` module expects an input of size $\left(N,C_{\text{in}}, L_{\text{in}}\right)$ or $\left(C_{\text{in}}, L_{\text{in}}\right)$ to produce an output of size $\left(N,C_{\text{out}}, L_{\text{out}}\right)$ or $\left(C_{\text{out}}, L_{\text{out}}\right)$ performing the following operation

$$
\begin{equation}
\text{out}\left(N_i, C_{\text{out}_j}\right) = \text{bias}\left(C_{\text{out}_j}\right) + \sum\limits_{k=0}^{C_{\text{in}}-1}\text{weight}\left(C_{\text{out}_j}, k\right) \star \text{input}\left(N_i, k\right)
\end{equation}
$$

Where

* $N$ is the batch size.
* $C_{\text{in}}$ is the number of input channels.
* $C_{\text{out}}$ is the number of output channels.
* $L_{\text{in}}$ is the length of the input tensor (i.e. `x.size(-1)` assuming an input tensor `x`).
* $L_{\text{out}}$ is the length of the output tensor (i.e. `y.size(-1)` assuming an output tensor `y`).
* $\star$ is the cross-correlation operator.

Additionally, $L_{\text{out}}$ (`y.size(-1)`) will depend on $L_{\text{in}}$ (`x.size(-1)`), `padding`, `dilation`, `kernel_size` and `stride` parameters. The relationship between these can be expressed as

$$
\begin{equation}
L_\text{out}=\left[\frac{L_{\text{in}}+2\times\text{padding} - \text{dilation}\times\left(\text{kernel\_size} - 1\right) - 1}{\text{stride}}+1\right]
\end{equation}
$$

## Complexity

### Number of filters
In order to calculate the number of operations performed this module, it is necessary to understand the impact of the `groups` parameter on the overall complexity, and the number of filters $\psi$ a network instance will have based on this. According to <a href="https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html" target="_blank">the official `torch.nn.Conv1d` documentation</a>

> `groups` controls the connections between inputs and outputs. `in_channels` and `out_channels` must both be divisible by `groups`.

> For example:
> At `groups=1,` all inputs are convolved to all outputs.
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

Now the [number of filters](#number-of-filters) $\psi$ are known, it is necessary to compute how many operations each filter performs. As shown in [Figure 1](#conv1d-kernel-diagram), for each kernel position there will be $\text{kernel\_size}$ multiplications (i.e. each kernel element multiplied by a slice of the input tensor of the same size) and $\text{kernel\_size}-1$ additions to aggregate the result and obtain one element of the output.

<figure markdown="span" id="conv1d-kernel-diagram">
  ![Image title](/img/conv1d-kernel-diagram-light.svg#only-light){ width="600" }
  ![Image title](/img/conv1d-kernel-diagram-dark.svg#only-dark){ width="600" }
  <figcaption>Figure 1. Operations per kernel position to obtain the output tensor.</figcaption>
</figure>

Since each element in $L_\text{out}$ is the result of the operations carried out for a single kernel position, the number of operations per filter $\lambda$ can be expressed as

$$
\begin{equation}
\lambda=L_{\text{out}}\times\left(\text{kernel\_size}+\left(\text{kernel\_size}-1\right)\right)
\end{equation}
$$

!!! note
    Please note that the batch size $N$ will be ignored for now, but it will be included later on.

### Filter aggregation
Now that the [number of filters](#number-of-filters) and the number of [operations per filter](#operations-per-filter) are known, it is necessary compute the operations needed to aggregate each group of filters $\gamma$ to produce each output channel $C_\text{out}$. These operations correspond to simple element-wise additions and can be expressed as

$$
\begin{equation}
\gamma=C_{\text{out}}\times L_\text{out}\times\left(\left(\frac{C_{\text{in}}}{\text{groups}}-1\right)+1\right)
\end{equation}
$$

Where the term $\left(\frac{C_{\text{in}}}{\text{groups}}-1\right)$ corresponds to the number of grouped connections between input and outputs channels $\frac{C_{\text{in}}}{\text{groups}}$, subtracted by $1$ because the operation is an addition. The $L_\text{out}$ factor accounts for the number of elements per filters, and $C_{\text{out}}$ expand this calculation to all output channels. Finally, the remaining $+1$ corresponds to the bias term $b$ that was not included so far, and that is added to each resulting output channel element. Note that this last term is only added if the module is instantiated using `bias=True`.

$$
\begin{equation}
\gamma=\begin{cases}
    C_{\text{out}}\times L_\text{out}\times\left(\frac{C_{\text{in}}}{\text{groups}}\right), & \text{if}\ \text{bias}=\text{True} \\
    C_{\text{out}}\times L_\text{out}\times\left(\frac{C_{\text{in}}}{\text{groups}}-1\right), & \text{if}\ \text{bias}=\text{False}
\end{cases}
\end{equation}
$$

Now putting together all different factors that contribute to the total number of operations $\phi$ as well including the batch size $N$, the total number of operations $\phi$ can be summarized as

$$
\begin{equation}
\phi=N\times\left(\psi\times\lambda+\gamma\right)
\end{equation}
$$

Where

* $N$ is the batch size.
* $\psi$ is the [number of filters](#number-of-filters).
* $\lambda$ is the number of [operations per filter](#operations-per-filter).
* $\gamma$ is the number of [filter aggregation](#filter-aggregation) operations.

## Summary