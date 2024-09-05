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
* $H_{\text{in}}$ is the height of the input tensor (i.e. `x.size(-2)` assuming an input tensor `x`)
* $W_{\text{in}}$ is the width of the input tensor (i.e. `x.size(-1)` assuming an input tensor `x`)
* $H_{\text{out}}$ is the height of the output tensor (i.e. `y.size(-2)` assuming an output tensor `y`)
* $W_{\text{out}}$ is the width of the output tensor (i.e. `y.size(-1)` assuming an output tensor `y`)

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

### Operations pr filter


### Filter aggregation
Now that the [number of filters](#number-of-filters) and the number of [operations per filter](#operations-per-filter) are known, it is necessary compute the operations needed to aggregate each group of filters $\gamma$ to produce each output channel $C_{out}$. These operations correspond to simple element-wise additions and can be expressed as


### Total operations


## Summary