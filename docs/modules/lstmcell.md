# LSTMCell (`torch.nn.LSTMCell`)
A `torch.nn.LSTMCell` correspond to a single cell of a Long Short-Term Memory Layer (`torch.nn.LSTM`). A `torch.nn.LSTMCell` takes in an **input** $x$, a **hidden state** $h$ and a **cell state** $c$ . Internally, it has an **input gate** $i$, a **forget gate** $f$, a **cell gate** $g$ and an **output gate** $o$ that help to propagate information between time steps. These are combined to generate the `torch.nn.LSTMCell` outputs. The relationship between these tensors is the following:

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
* $\odot$ is the <a href="https://en.wikipedia.org/wiki/Hadamard_product_(matrices)" target="_blank">Hadamard product</a> or element-wise product.
* $b_{ii}$, $b_{hi}$, $b_{if}$, $b_{hf}$, $b_{ig}$, $b_{hg}$, $b_{io}$ and $b_{ho}$ are bias tensors of size $\left(H_{out}\right)$.

## Complexity
In order to compute the complexity of a single `torch.nn.LSTMCell`, it is necessary to estimate the number of operations of all six aforementioned equations. For the sake of simplicity, for operations involving sigmoid and hyperbolic tangent, the aforementioned equations will be used and exponentials will be counted as a single operation.

!!! note
    During the following operations, some tensors have to be transposed in order to have compatible dimensions to perform matrix multiplication, even thought this is not explicitly mentioned in <a href="https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html", target="_blank">PyTorch `torch.nn.LSTMCell’s` documentation</a>. Additionally, some weight tensors are stacked. For instance, $W_{ii}$, $W_{if}$ $W_{ig}$ and $W_{io}$ are implemented as a single tensor of size $\left(4\times H_{out},H_{in} \right)$, and $W_{hi}$, $W_{hf}$, $W_{hg}$ and $W_{ho}$ are implemented as a single tensor of size $\left(4\times H_{out},H_{out} \right)$, possibly due to efficiency reasons.

### Input gate
The tensor sizes involved in the operations performed to calculate the input gate $i$ are

$$
\begin{align}
i = \sigma\Bigg(\underbrace{\left(H_{out}, H_{in}\right) \times \left(N, H_{in}\right)}_{W_{ii}x} + \underbrace{H_{out}}_{b_{ii}} + \underbrace{\left(H_{out}, H_{out}\right) \times \left(N, H_{out}\right)}_{W_{hi}h} + \underbrace{H_{out}}_{b_{hi}}\Bigg)
\end{align}
$$

In this case, $x$ (with shape $\left(N, H_{in}\right)$) has to be transposed. Additionally, $b$ will be implicitly broadcasted to be able to be summed with the tensor multiplication results. Then, the unwrapped and transposed shapes involved in the operations are

$$
\begin{align}
i &= \sigma\left(\left(H_{out}, H_{in}\right)\times\left(H_{in}, N\right)+\left(H_{out}, N\right)+\left(H_{out}, H_{out}\right)\times\left(H_{out}, N\right)+\left(H_{out}, N\right)\right)
\end{align}
$$

This will result in a tensor of shape $\left(H_{out}, N\right)$. To estimate the complexity of this operation, it is possible to reuse the results from [`torch.nn.Linear`](./linear.md) for both matrix multiplications and add the sigmoid operations $\sigma$. $i_{ops}$ (the operations of the input gate $i$) can be then broken down into four parts:

1. The operations to needed compute $W_{ii}x+b_{ii}$.
2. The operations needed to compute $W_{hi}h+b_{hi}$.
3. The operations needed to sum both results.
4. The operations needed to compute the sigmoid function $\sigma$ over this result.

For simplicity sake, the following definitions will be used:

$$
\begin{align}
i &= \sigma\left(\underbrace{W_{ii}x^T+b_{ii}}_{i_0}+\underbrace{W_{hi}h^T+b_{hi}}_{i_1}\right)
\end{align}
$$

Then, in terms of operations ($ops$) when `bias=True`

$$
\begin{align}
    i_{0_{ops}} &=\left(W_{ii}x^T+b_{ii}\right)_{ops} = 2\times N\times H_{out}\times H_{in} \\
    i_{1_{ops}} &=\left(W_{hi}x^T+b_{hi}\right)_{ops} = 2\times N\times H_{out}^2 \\
    \left(i_0+i_1\right)_{ops} &= N\times H_{out} \\
    \sigma_{ops} &= 3\times N\times H_{out} \\
    i_{ops} &= 2\times N\times H_{out}\left(2+H_{in}+ H_{out}\right)
\end{align}
$$

and when `bias=False`

$$
\begin{align}
    i_{0_{ops}} &=\left(W_{ii}x^T+b_{ii}\right)_{ops} = N\times H_{out}\times \left(2\times H_{in}-1\right) \\
    i_{i_{ops}}
    &=\left(W_{hi}x^T+b_{hi}\right)_{ops}=N\times H_{out}\times\left(2\times H_{out}-1\right) \\
    \left(i_0+i_1\right)_{ops} &= N\times H_{out} \\
    \sigma_{ops} &= 3\times N\times H_{out}\\
    i_{ops} &= 2\times N\times H_{out}\times\left(1+H_{in}+ H_{out}\right)
\end{align}
$$

### Forget and output gates
Since the dimensions of these gates are the same as the input gate $i$, it is trivial to observe that

$$
\begin{equation}
    i_{ops}=f_{ops}=o_{ops}
\end{equation}
$$

### Cell gate
The argument of the $\text{tanh}$ function has the same shape as the previously computed gates, yet the complexity of this function itself is the only difference between this gate and the others, then

$$
\begin{align}
    g_{ops}&=i_{0_{ops}}+i_{1_{ops}}+
    \left(i_0+i_1\right)_{ops}+\text{tanh}_{ops}\\
    \text{tanh}_{ops}&=7\times N\times H_{out}\\
\end{align}
$$

Replacing by the previously calculated results

$$
\begin{equation}
g_{ops}=\begin{cases}
    2\times N\times H_{out}\times \left(4+H_{in}+ H_{out}\right), & \text{if}\ \text{bias}=\text{True} \\
    2\times N\times H_{out}\times \left(3+H_{in} + H_{out}\right), &\text{if}\ \text{bias}=\text{False}
\end{cases}
\end{equation}
$$


### $c\prime$
The complexity of $c\prime$ corresponds to three element-wise operations between elements with shape $\left(H_{out}, N\right)$. Therefore its complexity is

$$
\begin{align}
    c\prime &= f\odot c+i\odot g\\
    c\prime_{ops} &= 3\times N\times H_{out}
\end{align}
$$

### $h\prime$
The complexity of $h\prime$ corresponds to one element-wise operation and a $\text{tanh}$ operation

$$
\begin{align}
    h\prime &= o \odot\text{tanh}\left(c\prime\right)\\
    \text{tanh}_{ops}&=7\times N\times H_{out}\\
    h\prime_{ops} &= 8\times N\times H_{out}
\end{align}
$$

### Total complexity
Finally, the total complexity is the sum of all individual contributions

$$
\begin{align}
    \text{LSTMCell}_{ops}=i_{ops}+f_{ops}+g_{ops}+o_{ops}+c\prime_{ops}+h\prime_{ops}
\end{align}
$$

In the case of `bias=True`, the total number of operations is

$$
\begin{align}
    \text{LSTMCell}_{ops} &= \underbrace{6 \times N \times H_{out}\times(2+H_{in} + H_{out})}_{i_{ops} + f_{ops} + o_{ops}} \nonumber \\
    &\quad+ \underbrace{2 \times N \times H_{out} \times (4 + H_{in} + H_{out})}_{g_{ops}} \nonumber \\
    &\quad+ \underbrace{11 \times N \times H_{\text{out}}}_{c\prime_{ops}+h\prime_{ops}} \\
    \text{LSTMCell}_{ops} &= 8\times N\times H_{out}\times\left( H_{in}+H_{out}+3.875\right)
\end{align}
$$

and for `bias=False`

$$
\begin{align}
    \text{LSTMCell}_{ops} &= \underbrace{6 \times N \times H_{out}\times(1+H_{in} + H_{out})}_{i_{ops} + f_{ops} + o_{ops}} \nonumber \\
    &\quad+ \underbrace{2 \times N \times H_{out} \times (3 + H_{in} + H_{out})}_{g_{ops}} \nonumber \\
    &\quad+ \underbrace{11 \times N \times H_{\text{out}}}_{c\prime_{ops}+h\prime_{ops}} \\
    \text{LSTMCell}_{ops} &= 8\times N\times H_{out}\times\left(H_{in}+H_{out}+2.875\right)
\end{align}
$$

## Summary
The number of operations $\phi$ operformed by a `torch.nn.LSTMCell` module can be estimated as

!!! success ""

    === "If `bias=True`"
        $\text{LSTMCell}_{ops} = 8\times N\times H_{out}\times\left( H_{in}+H_{out}+3.875\right)$

    === "If `bias=False`"
        $\text{LSTMCell}_{ops} = 8\times N\times H_{out}\times\left(H_{in}+H_{out}+2.875\right)$


Where

* $N$ is the batch size.
* $H_\text{in}$ is the number of input features.
* $H_\text{out}$ is the number of output features.