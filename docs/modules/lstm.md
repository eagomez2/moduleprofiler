# LSTM (`torch.nn.LSTM`)
A `torch.nn.LSTM` corresponds to a Long-short Term Memory module. That is - in essence - an arrangement or `torch.nn.LSTMCell` that can process an input tensor containing a sequence of steps and can use cell arrangements of configurable depth. The equations that rule a `torch.nn.LSTM` are the same as `torch.nn.LSTMCell`, except for the sequence length and the number of layers. Differently from a single `torch.nn.LSTMCell`, a `torch.nn.LSTM` has a **hidden state** per sequence step ($h_t$), an **input gate** per sequence step ($i_t$), a **forget gate** per sequence step ($f_t$), a **cell gate** per sequence step ($g_t$), and an **output gate** per sequence step ($o_t$), thus, there is also a $c$ tensor per time step ($c_t$), and an $h$ tensor per time step ($h_t$). Please note that the current step **hidden state** $h_t$ depends on the previous step hidden state $h_{t-1}$.

$$
\begin{align}
    i_t &= \sigma\left(W_{ii}x_t+b_{ii}+W_{hi}h_{\left(t-1\right)}+b_{hi}\right) \\
    f_t &= \sigma\left(W_{if}x_t+b_{if}+W_{hf}h_{\left(t-1\right)}+b_{hf}\right) \\
    g_t &= \text{tanh}\left(W_{ig}x_t+b_{ig}+W_{hg}h_{\left(t-1\right)}+b_{hg}\right) \\
    o_t &= \sigma\left(W_{io}x_t+b_{io}+W_{ho}h_{\left(t-1\right)}+b_{ho}\right) \\
    c_t &= f_t\odot c_{\left(t-1\right)}+i_t\odot g_t \\
    h_t &= o_t \odot\text{tanh}\left(c_t\right)
\end{align}
$$

Where

* $x$ is the input tensor of size $\left(L, H_{in}\right)$ or $\left(L, N, H_{in}\right)$ when `batch_first=True`, or $\left(N, L, H_{in}\right)$ when `batch_first=True`.
* $h_t$ is the hidden state tensor at sequence step $t$ of size $\left(N, H_{out}\right)$ or $\left(H_{out}\right)$.
* $H_{in}$ and $H_{out}$ are the number of input and output features, respectively.
* $L$ is the sequence length.
* $N$ is the batch size.
* $c$ is the cell state tensor of size $\left(N, H_{out}\right)$ or $\left(H_{out}\right)$.
* $W_{ii}$, $W_{if}$, $W_{ig}$ and $W_{io}$ are weight tensors of size $\left(H_{out}, H_{in}\right)$ in the first layer and $\left(H_{out}, D\times H_{out}\right)$ in subsequent layers. 
* $W_{hi}$, $W_{hf}$, $W_{hg}$ and $W_{ho}$ are weight tensors of size $\left(H_{out}, H_{out}\right)$.
* $\sigma$ is the sigmoid function and can be defined as $\sigma\left(x\right)=\frac{1}{1+e^{-x}}$.
* $\text{tanh}$ is the hyperbolic tangent function and can be defined as $\text{tanh}\left(x\right)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$.
* $\odot$ is the <a href="https://en.wikipedia.org/wiki/Hadamard_product_(matrices)" target="_blank">Hadamard product</a> or element-wise product.
* $b_{ii}$, $b_{hi}$, $b_{if}$, $b_{hf}$, $b_{ig}$, $b_{hg}$, $b_{io}$ and $b_{ho}$ are bias tensors of size $\left(H_{out}\right)$.

!!! note
    Please note that some weight tensor sizes may differ from <a href="https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html" target="_blank">Pytorch `torch.nn.LSTM`'s documentation</a> due to the fact that some tensors are stacked. For instance, $W_{ii}$, $W_{if}$, $W_{ig}$ and $W_{io}$ tensors of each layer are implemented as a single tensor of size $\left(4\times H_{out}, H_{in}\right)$ for the first layer, and $\left(4\times H_{out}, D\times H_{out}\right)$ for subsequent layers. Similarly $W_{hi}$, $W_{hf}$, $W_{hg}$ and $W_{ho}$ are implemented as a single tensor of size $\left(4\times H_{out}, H_{out}\right)$. The number of layers is controlled by the `num_layers` parameter, and the number of directions $D$ is controlled by the `birectional` parameter.

!!! note
    The complexity of the `dropout` parameter is not considered in the following calculations, since it is usually temporarily used during training and then disabled during inference.

## Complexity
It is possible to reuse the calculation for `torch.nn.GRUCell` to estimate the complexity of `torch.nn.GRU`. However, there are a couple of additional considerations. First, when `num_layers > 1`, the second layer takes the output(s) of the first layer as input. This means that $W_{ii}$, $W_{if}$, $W_{ig}$ and $W_{io}$ will have size $\left(H_{out}, H_{out}\right)$ if `bidirectional=False` and size $\left(H_{out}, 2\times H_{out}\right)$ if `bidirectional=True`. Secondly and differently from `torch.nn.LSTMCell`, `torch.nn.LSTM` can process an input containing bigger sequence lenghts, therefore the same calculations estimated before will repeat $L$ times where $L$ sequence length.

!!! warning
    Please review the [`torch.nn.LSTMCell` complexity documentation](./lstmcell.md) before continuing, as the subsequent sections will reference formulas from that layer without re-deriving them.

### Unidirectional
The complexity of the first layer is the same as as `torch.nn.LSTMCell` that if `bias=True`can be simplified to

$$
\begin{align}
    \text{LSTM}_{ops}|_{\text{layer}=0} = 8\times N \times H_{out}\times\left(H_{in}+H_{out}+3.875\right)
\end{align}
$$

and when `bias=False`

$$
\begin{align}
    \text{LSTM}_{ops}|_{\text{layer}=0} = 8\times N \times H_{out}\times\left(H_{in}+H_{out}+2.875\right)
\end{align}
$$

For subsequent layers it is necessary to replace $H_{in}$ by $H_{out}$, then when `bias=True`

$$
\begin{align}
    \text{LSTM}_{ops}|_{\text{layer}\geq 1} = 8\times N \times H_{out}\times\left(2\times H_{out}+3.875\right)
\end{align}
$$

#### Total complexity
Now it is necessary to include the sequence length $L$ in the input tensor $x$ to obtain the total complexity, since the previous calculation will be repeatead $L$ times. The total complexity for `bidirectional=False` is

$$
\begin{align}
    \text{LSTM}_{ops} = L\times\left(\text{LSTM}_{ops}|_{\text{layer}=0} + \left(\text{num\_layers} - 1\right)\times \text{LSTM}_{ops}|_{\text{layer}\geq 1}\right)
\end{align}
$$

When `bias=True` this expression becomes

$$
\begin{align}
    \text{LSTM}_{ops} &= L\times \underbrace{8\times N \times H_{out}\times\left(H_{in}+H_{out}+3.875\right)}_{\text{LSTM}_{ops}|_{\text{layer}=0}} \nonumber \\
    &\quad + L\times \left(\text{num\_layers} - 1 \right)\times \underbrace{8\times N \times H_{out}\times\left(2\times H_{out}+3.875\right)}_{\text{LSTM}_{ops}|_{\text{layer}\geq 1}} \nonumber \\
    \text{LSTM}_{ops} &= 8\times L \times N \times H_{out}\times \left(H_{in}+\left(2\times\text{num\_layers}-1\right)\times H_{out}+3.875\times\text{num\_layers}\right)
\end{align}
$$

and when `bias=False`

$$
\begin{align}
    \text{LSTM}_{ops} &= L\times \underbrace{8\times N \times H_{out}\times\left(H_{in}+H_{out}+2.875\right)}_{\text{LSTM}_{ops}|_{\text{layer}=0}} \nonumber \\
    &\quad + L\times \left(\text{num\_layers} - 1 \right)\times \underbrace{8\times N \times H_{out}\times\left(2\times H_{out}+2.875\right)}_{\text{LSTM}_{ops}|_{\text{layer}\geq 1}} \nonumber \\
    \text{LSTM}_{ops} &= 8\times L \times N \times H_{out}\times \left(H_{in}+\left(2\times\text{num\_layers}-1\right)\times H_{out}+2.875\times\text{num\_layers}\right)
\end{align}
$$

### Bidirectional
For the case of `bidirectional=True` the same considerations explained [at the beginning of this section](#complexity) should be taken into account. Additionally, each cell will approximately duplicate its calculations because one subset of the output is calculated using the forward direction of the input sequence $x$, and the remaining one uses the reverse input sequence $x$. Please note that each direction of the input sequence will have its own set of weights. Finally, both outputs will be concatenated to produce a tensor of size $\left(L, N, D\times H_{out}\right)$ with $D=2$ in this case. When `num_layers > 1`, this is also the size of the input size for layers after the first one.

The complexity of the first layer when `bidirectional=True` and `bias=True` is

$$
\begin{align}
    \text{LSTM}_{ops}|_{\text{layer}=0} = 16\times N \times H_{out}\times\left(H_{in}+H_{out}+3.875\right)
\end{align}
$$

and when `bias=False`

$$
\begin{align}
    \text{LSTM}_{ops}|_{\text{layer}=0} = 16\times N \times H_{out}\times\left(H_{in}+H_{out}+2.875\right)
\end{align}
$$

For subsequent layers it is necessary to replace $H_{in}$ by $2\times H_{out}$. Then when `bias=True`

$$
\begin{align}
    \text{LSTM}_{ops}|_{\text{layer}\geq 1} = 16\times N \times H_{out}\times\left(3\times H_{out}+3.875\right)
\end{align}
$$

and when `bias=False`

$$
\begin{align}
    \text{LSTM}_{ops}|_{\text{layer}\geq 1} = 16\times N \times H_{out}\times\left(3\times H_{out}+2.875\right)
\end{align}
$$

#### Total complexity
Now it is necessary to include the sequence length $L$ in the input tensor $x$ to obtain the total complexity, since the previous calculation will be repeatead $L$ times. The total complexity for `bidirectional=True` is

$$
\begin{align}
    \text{LSTM}_{ops} = L\times\left(\text{LSTM}_{ops}|_{\text{layer}=0} + \left(\text{num\_layers} - 1\right)\times \text{LSTM}_{ops}|_{\text{layer}\geq 1}\right)
\end{align}
$$

When `bias=True` this expression becomes

$$
\begin{align}
    \text{LSTM}_{ops} &= L\times\underbrace{16\times N \times H_{out}\times\left(H_{in}+H_{out}+3.875\right)}_{\text{LSTM}_{ops}|_{\text{layer}=0}} \nonumber \\
    &\quad + L\times\left(\text{num\_layers} - 1 \right)\times \underbrace{\left(16\times N \times H_{out}\times\left(3\times H_{out}+3.875\right)\right)}_{\text{GRU}_{ops}|_{\text{layer}\geq 1}} \nonumber \\
    \text{LSTM}_{ops} &= 16\times L\times N \times H_{out}\times \left(H_{in}+\left(3\times\text{num\_layers}-2\right)\times H_{out}+3.875\times\text{num\_layers}\right)
\end{align}
$$

and when `bias=False`

$$
\begin{align}
    \text{LSTM}_{ops} &= L\times\underbrace{16\times N \times H_{out}\times\left(H_{in}+H_{out}+2.875\right)}_{\text{LSTM}_{ops}|_{\text{layer}=0}} \nonumber \\
    &\quad + L\times\left(\text{num\_layers} - 1 \right)\times \underbrace{\left(16\times N \times H_{out}\times\left(3\times H_{out}+2.875\right)\right)}_{\text{GRU}_{ops}|_{\text{layer}\geq 1}} \nonumber \\
    \text{LSTM}_{ops} &= 16\times L\times N \times H_{out}\times \left(H_{in}+\left(3\times\text{num\_layers}-2\right)\times H_{out}+2.875\times\text{num\_layers}\right)
\end{align}
$$

## Summary
The number of operations performed by a `torch.nn.LSTM` module can be estimated as

!!! success ""
    === "If `bias=True` and `bidirectional=False`"
        $\text{LSTM}_{ops} = 8\times L\times N \times H_{out}\times \left(H_{in}+\left(2\times\text{num\_layers}-1\right)\times H_{out}+3.875\times\text{num\_layers}\right)$
    
    === "If `bias=False` and `bidirectional=False`"
        $\text{LSTM}_{ops} = 8\times L\times N \times H_{out}\times \left(H_{in}+\left(2\times\text{num\_layers}-1\right)\times H_{out}+2.875\times\text{num\_layers}\right)$
    
    === "If `bias=True` and `bidirectional=True`" 
        $\text{LSTM}_{ops} = 16\times L\times N \times H_{out}\times \left(H_{in}+\left(3\times\text{num\_layers}-2\right)\times H_{out}+3.875\times\text{num\_layers}\right)$

    === "If `bias=False` and `bidirectional=True`"
        $\text{LSTM}_{ops} = 16\times L\times N \times H_{out}\times \left(H_{in}+\left(3\times\text{num\_layers}-2\right)\times H_{out}+2.875\times\text{num\_layers}\right)$
    
Where 

* $L$ is the sequence length.
* $N$ is the batch size.
* $H_{in}$ and $H_{out}$ are the number of input and output features, respectively.
* $\text{num\_layers}$ is the number of layers.