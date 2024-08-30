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
    Please note that some weight tensor shapes may differ from <a href="https://pytorch.org/docs/stable/generated/torch.nn.GRU.html" target="_blank">Pytorch `torch.nn.GRU`'s documentation</a> due to the fact that some tensors are stacked. For instance, $W_{ir}$, $W_{iz}$ and $W_{in}$ tensors of each layer are implemented as a single tensor of size $\left(3\times H_{out}, H_{in}\right)$ for the first layer, and $\left(3\times H_{out}, D\times H_{out}\right)$ for subsequent layers. Similarly $W_{hr}$, $W_{hz}$ and $W_{hn}$ are implemented as a single tensor of size $\left(3\times H_{out}, H_{out}\right)$. The number of layers is controlled by the `num_layers` parameter, and the number of directions $D$ is controlled by the `birectional` parameter.

!!! note
    The complexity of the `dropout` parameter is not considered in the following calculations, since it is usually temporarily used during training and then disabled during inference.

## Complexity
It is possible to reuse the calculation for `torch.nn.GRUCell` to estimate the complexity of `torch.nn.GRU`. However, there are a couple of additional considerations. First, when `num_layers > 1`, the second layer takes the output(s) of the first layer as input. This means that $W_{ir}$, $W_{iz}$ and $W_{in}$ will have size $\left(H_\text{out}, H_\text{out}\right)$ if `bidirectional=False` and size $\left(H_\text{out}, 2\times H_\text{out}\right)$ if `bidirectional=True`.

!!! warning
    Please review the [`torch.nn.GRUCell` complexity documentation](./grucell.md) before continuing, as the subsequent sections will reference formulas from that layer without re-deriving them.

### Unidirectional
The complexity of the first layer is the same as as `torch.nn.GRUCell` that if `bias=True`can be simplified to

$$
\begin{align}
    \text{GRU}_{ops}|_{\text{layer}=0} = 6\times N \times H_{out}\times\left(H_{in}+H_{out}+3.5\right)
\end{align}
$$

and when `bias=False`

$$
\begin{align}
    \text{GRU}_{ops}|_{\text{layer}=0} = 6\times N \times H_{out}\times\left(H_{in}+H_{out}+2.5\right)
\end{align}
$$

For subsequent layers it is necessary to replace $H_{in}$ by $H_{out}$, then when `bias=True`

$$
\begin{align}
    \text{GRU}_{ops}|_{\text{layer}\geq 1} = 6\times N \times H_{out}\times\left(2\times H_{out}+3.5\right)
\end{align}
$$

and when `bias=False`

$$
\begin{align}
    \text{GRU}_{ops}|_{\text{layer}\geq 1} = 6\times N \times H_{out}\times\left(2\times H_{out}+2.5\right)
\end{align}
$$


#### Total complexity
The total complexity for `bidirectional=False` is

$$
\begin{align}
    \text{GRU}_{ops} &= \text{GRU}_{ops}|_{\text{layer}=0} + \left(\text{num\_layers} - 1 \right)\times \text{GRU}_{ops}|_{\text{layer}\geq 1}
\end{align}
$$

When `bias=True` this expression becomes

$$
\begin{align}
    \text{GRU}_{ops} &= \underbrace{6\times N \times H_{out}\times\left(H_{in}+H_{out}+3.5\right)}_{\text{GRU}_{ops}|_{\text{layer}=0}} \nonumber \\
    &\quad + \left(\text{num\_layers} - 1 \right)\times \underbrace{\left(6\times N \times H_{out}\times\left(2\times H_{out}+3.5\right)\right)}_{\text{GRU}_{ops}|_{\text{layer}\geq 1}} \nonumber \\
    \text{GRU}_{ops} &= 6\times N \times H_{out}\times \left(H_{in}+\left(2\times\text{num\_layers}-1\right)\times H_{out}+3.5\times\text{num\_layers}\right)
\end{align}
$$

and when `bias=False`

$$
\begin{align}
    \text{GRU}_{ops} &= \underbrace{6\times N \times H_{out}\times\left(H_{in}+H_{out}+2.5\right)}_{\text{GRU}_{ops}|_{\text{layer}=0}} \nonumber \\
    &\quad + \left(\text{num\_layers} - 1 \right)\times \underbrace{\left(6\times N \times H_{out}\times\left(2\times H_{out}+2.5\right)\right)}_{\text{GRU}_{ops}|_{\text{layer}\geq 1}} \nonumber \\
    \text{GRU}_{ops} &= 6\times N \times H_{out}\times \left(H_{in}+\left(2\times\text{num\_layers}-1\right)\times H_{out}+2.5\times\text{num\_layers}\right)
\end{align}
$$

### Bidirectional
For the case of `bidirectional=True` the same considerations explained [at the beginning of this section](#complexity) should be taken into account. Additionally, each cell will approximately duplicate its calculations because one subset of the output is calculated using the forward direction of the input sequence $x$, and the remaining one uses the reverse input sequence $x$. Please note that each direction of the input sequence will have its own set of weights, even though <a href="https://github.com/pytorch/pytorch/issues/59332" target="blank">this is not documented at the moment of writing this documentation</a>. Finally, both outputs will be concatenated to produce a tensor of size $\left(L, N, D\times H_{out}\right)$ with $D=2$ in this case. When `num_layers > 1`, this is also the shape of the input size for layers after the first one.

The complexity of the first layer when `bidirectional=True` and `bias=True` is

$$
\begin{align}
    \text{GRU}_{ops}|_{\text{layer}=0} = 12\times N \times H_{out}\times\left(H_{in}+H_{out}+3.5\right)
\end{align}
$$

and when `bias=False`

$$
\begin{align}
    \text{GRU}_{ops}|_{\text{layer}=0} = 12\times N \times H_{out}\times\left(H_{in}+H_{out}+2.5\right)
\end{align}
$$

For subsequent layers it is necessary to replace $H_{in}$ by $2\times H_{out}$. Then when `bias=True`

$$
\begin{align}
    \text{GRU}_{ops}|_{\text{layer}\geq 1} = 12\times N \times H_{out}\times\left(3\times H_{out}+3.5\right)
\end{align}
$$

and when `bias=False`

$$
\begin{align}
    \text{GRU}_{ops}|_{\text{layer}\geq 1} = 12\times N \times H_{out}\times\left(3\times H_{out}+2.5\right)
\end{align}
$$

#### Total complexity
The total complexity for `bidirectional=True` is

$$
\begin{align}
    \text{GRU}_{ops} &= \text{GRU}_{ops}|_{\text{layer}=0} + \left(\text{num\_layers} - 1 \right)\times \text{GRU}_{ops}|_{\text{layer}\geq 1}
\end{align}
$$

When `bias=True` this expression becomes

$$
\begin{align}
    \text{GRU}_{ops} &= \underbrace{12\times N \times H_{out}\times\left(H_{in}+H_{out}+3.5\right)}_{\text{GRU}_{ops}|_{\text{layer}=0}} \nonumber \\
    &\quad + \left(\text{num\_layers} - 1 \right)\times \underbrace{\left(12\times N \times H_{out}\times\left(3\times H_{out}+3.5\right)\right)}_{\text{GRU}_{ops}|_{\text{layer}\geq 1}} \nonumber \\
    \text{GRU}_{ops} &= 12\times N \times H_{out}\times \left(H_{in}+\left(3\times\text{num\_layers}-2\right)\times H_{out}+3.5\times\text{num\_layers}\right)
\end{align}
$$

and when `bias=False`

$$
\begin{align}
    \text{GRU}_{ops} &= \underbrace{12\times N \times H_{out}\times\left(H_{in}+H_{out}+2.5\right)}_{\text{GRU}_{ops}|_{\text{layer}=0}} \nonumber \\
    &\quad + \left(\text{num\_layers} - 1 \right)\times \underbrace{\left(12\times N \times H_{out}\times\left(3\times H_{out}+2.5\right)\right)}_{\text{GRU}_{ops}|_{\text{layer}\geq 1}} \nonumber \\
    \text{GRU}_{ops} &= 12\times N \times H_{out}\times \left(H_{in}+\left(3\times\text{num\_layers}-2\right)\times H_{out}+2.5\times\text{num\_layers}\right)
\end{align}
$$

## Summary
The number of operations performed by a `torch.nn.GRU` module can be estimated as

!!! success ""
    === "If `bias=True` and `bidirectional=False`"
        $\text{GRU}_{ops} = 6\times N \times H_{out}\times \left(H_{in}+\left(2\times\text{num\_layers}-1\right)\times H_{out}+3.5\times\text{num\_layers}\right)$
    
    === "If `bias=False` and `bidirectional=False`"
        $\text{GRU}_{ops} = 6\times N \times H_{out}\times \left(H_{in}+\left(2\times\text{num\_layers}-1\right)\times H_{out}+2.5\times\text{num\_layers}\right)$
    
    === "If `bias=True` and `bidirectional=True`" 
        $\text{GRU}_{ops} = 12\times N \times H_{out}\times \left(H_{in}+\left(3\times\text{num\_layers}-2\right)\times H_{out}+3.5\times\text{num\_layers}\right)$

    === "If `bias=False` and `bidirectional=True`"
        $\text{GRU}_{ops} = 12\times N \times H_{out}\times \left(H_{in}+\left(3\times\text{num\_layers}-2\right)\times H_{out}+2.5\times\text{num\_layers}\right)$
    
Where

* $N$ is the batch size.
* $H_\text{in}$ is the number of input features.
* $H_\text{out}$ is the number of output features.
* $\text{num\_layers}$ is the number of layers. When `num_layers > 1`, the output of the first layer is fed into the second one.