# GRU (`torch.nn.GRU`)
A `torch.nn.GRU` corresponds to a Gated Recurrent Unit. That is - in essence - an arrangement or `torch.nn.GRUCell` that can process an input tensor containing several steps and can use cell arrangements of configurable depth. The equations that rule a `torch.nn.GRU` are the same as `torch.nn.GRUCell`, except for the time steps and the number of layers. Differently from a single `torch.nn.GRUCell`, a `torch.nn.GRU` has a **hidden state** per time step ($h_t$), a **reset gate** per time step ($r_t$) and an **update gate** per time step ($z_t$), thus, there is also a $n$ tensor per time step ($n_t$). Please note that the current step **hidden state** $h_t$ depends on the previous step hidden state $h_{t-1}$.

$$
\begin{align}
    r_t &= \sigma\left(W_{ir}x_t+b_{ir}+W_{hr}h_{\left(t-1\right)}+b_{hr}\right) \\
    z_t &= \sigma\left(W_{iz}x_t+b_{iz}+W_{hz}h_{\left(t-1\right)}+b_{hz}\right) \\
    n_t &= \text{tanh}\left(W_{in}x_t+b_{in}+r_t\odot\left(W_{hn}h_{\left(t-1\right)}+b_{hn}\right)\right) \\
    h_t &= (1-z_t)\odot n_t+z_t\odot h_{\left(t-1\right)}
\end{align}
$$

Where

* $x$ is the input tensor of size $\left(L, H_{in}\right)$, $\left(L, N, H_{in}\right)$ when `batch_first=False` or $\left(N, L, H_{in}\right)$ when `batch_first=True`.
* $h_t$ is the hidden state tensor at time step $t$ of size $\left(N, H_{out}\right)$ or $\left(H_{out}\right)$.
* $W_{ir}$, $W_{iz}$ and $W_{in}$ are weight tensors of size $\left(H_{out}, H_{in}\right)$ in the first layer and $\left(H_{out}, D\times H_{out}\right)$ in subsequent layers.
* $W_{hr}$, $W_{hz}$ and $W_{hn}$ are weight tensors of size $\left(H_{out}, H_{out}\right)$ 
* $D$ is $2$ if `bidirectional=True` and $1$ if `bidirectional=False`.
* $\sigma$ is the sigmoid function and can be defined as $\sigma\left(x\right)=\frac{1}{1+e^{-x}}$.
* $\text{tanh}$ is the hyperbolic tangent function and can be defined as $\text{tanh}\left(x\right)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$.
* $\odot$ is the <a href="https://en.wikipedia.org/wiki/Hadamard_product_(matrices)" target="_blank">Hadamard product</a> or element-wise product.
* $b_{ir}$, $b_{iz}$, $b_{in}$, $b_{hr}$, $b_{hz}$ and $b_{hn}$ are bias tensors of size $\left(H_{out}\right)$.

!!! note
    Please note that some weight tensor shapes may differ from <a href="https://pytorch.org/docs/stable/generated/torch.nn.GRU.html" target="_blank">Pytorch `torch.nn.GRU`'s documentation</a> due to the fact that some tensors are stacked. For instance, $W_{ir}$, $W_{iz}$ and $W_{in}$ tensors of each layer are implemented as a single tensor of size $\left(3\times H_{out}, H_{in}\right)$ for the first layer, and $\left(3\times H_{out}, D\times H_{out}\right)$ for subsequent layers. Similarly $W_{hr}$, $W_{hz}$ and $W_{hn}$ are implemented as a single tensor of size $\left(3\times H_{out}, H_{out}\right)$. The tensors of all layers are then stored as a `list` of `torch.Tensor` objects. The number of layers is controlled by the `num_layers` parameter, and the number of directions $D$ is controlled by the `birectional` parameter.

!!! note
    The complexity of the `dropout` parameter is not considered in the following calculations, since it is usually temporarily used during training and then disabled during inference.

## Complexity

## Summary