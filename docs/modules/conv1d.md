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

## Summary