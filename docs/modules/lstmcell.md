# LSTMCell (`torch.nn.LSTMCell`)
A `torch.nn.LSTMCell` correspond to a single cell of a Long Short-Term Memory Layer (`torch.nn.LSTM`). A `torch.nn.LSTMCell` takes in an **input** $x$, a **hidden state** $h$ and a **cell state** $c$ . Internally, it has an **input gate** $i$, a **forget gate** $f$, a **cell gate** $g$ and an **output gate** $o$ that helps to propagate information between time steps. These are combined to generate the `torch.nn.LSTMCell` outputs. The relationship between these tensors is the following:

$$
\begin{align}
    i &= \sigma\left(W_{ii}x+b_{ii}+W_{hi}h+b_{hi}\right) \\
    f &= \sigma\left(W_{if}x+b_{if}+W_{hf}h+b_{hf}\right) \\
    g &= \text{tanh}\left(W_{ig}x+b_{ig}+W_{hg}h+b_{hg}\right) \\
    o &= \sigma\left(W_{io}x+b_{io}+W_{ho}h+b_{ho}\right) \\
    c\prime &= f\odot c+i\odot g \\
    h\prime &= o \odot\text{tanh}\left(c\prime\right)
\end{align}
$$

Where

* $x$ is the input tensor of size $\left(N, H_{in}\right)$ or $\left(H_{in}\right)$.
* $h$ is the hidden tensor of size $\left(N, H_{out}\right)$.
* $c$ is the cell state tensor of size $\left(N, H_{out}\right)$.
* $W_{ii}$, $W_{if}$, $W_{ig}$ and $W_{io}$ are weight tensors of size $\left(H_{out}, H_{in}\right)$. 
* $W_{hi}$, $W_{hf}$, $W_{hg}$ and $W_{ho}$ are weight tensors of size $\left(H_{out}, H_{out}\right)$.
* $\sigma$ is the sigmoid function and can be defined as $\sigma\left(x\right)=\frac{1}{1+e^{-x}}$.
* $\text{tanh}$ is the hyperbolic tangent function and can be defined as $\text{tanh}\left(x\right)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$.
* $\odot$ is the Hadamard product or element-wise product.
* $b_{ii}$, $b_{hi}$, $b_{if}$, $b_{hf}$, $b_{ig}$, $b_{hg}$, $b_{io}$ and $b_{ho}$ are bias tenbsors of size $\left(H_{out}\right)$.

## Complexity
In order to compute the complexity of a single `torch.nn.LSTMCell`, we just need to estimate the number of operations of all six aforementioned equations. For the sake of simplicity, for operations involving sigmoid and hyperbolic tangent, the listed equations will be used and exponentials will be counted as a single operation.

!!! note
    During the following operations, some tensors have to be transposed in order to have compatible dimensions to perform matrix multiplication, even thought this is not explicitly mentioned in <a href="https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html", target="_blank">PyTorch `torch.nn.LSTMCell’s` documentation</a>. Additionally, some weight tensors are stacked. For instance, $W_{ii}$, $W_{if}$ $W_{ig}$ and $W_{io}$ are implemented as a single tensor of size $\left(4\times H_{out},H_{in} \right)$, and $W_{hi}$, $W_{hf}$, $W_{hg}$ and $W_{ho}$ are implemented as a single tensor of size $\left(4\times H_{out},H_{out} \right)$, possibly due to efficiency reasons.

### Input gate
The tensor sizes involved in the operations performed to calculate the input gate $i$ are

$$
\begin{align}
i = \sigma\Bigg(\underbrace{\left(H_{out}, H_{in}\right) \times \left(N, H_{in}\right)}_{W_{ii}x} + \underbrace{H_{out}}_{b_{ii}} + \underbrace{\left(H_{out}, H_{out}\right) \times \left(N, H_{out}\right)}_{W_{hi}h} + \underbrace{H_{out}}_{b_{hi}}\Bigg)
\end{align}
$$

In this case, $x$ (with shape $\left(N, H_{in}\right)$) has to be transposed. Additionally, $b$ will be implicitly broadcasted to be able to be summed with the tensor multiplication results. Then, the unwrapped and transposed shapes involved in the operations are:

### Forget and output gates

### Cell gate

### $c\prime$

### $h\prime$

### Total complexity

## Summary