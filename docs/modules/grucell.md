# GRUCell (`torch.nn.GRUCell`)
A `torch.nn.GRUCell` corresponds to a single cell of a Grated Recurrent Unit (`torch.nn.GRU`). A `torch.nn.GRUCell` takes an **input** $x$, a **hidden state** $h$. Internally, it
has a **reset gate** $r$ and an **update gate** $z$ that help to propagate information between time steps. These are combined to generate $n$, that is then used to create a new hidden state $h\prime$. The relationship between these tensors is defines as

$$
\begin{align}
    r &= \sigma\left(W_{ir}x+b_{ir}+W_{hr}h+b_{hr}\right) \\
    z &= \sigma\left(W_{iz}x+b_{iz}+W_{hz}h+b_{hz}\right) \\
    n &= \text{tanh}\left(W_{in}x+b_{in}+r\odot\left(W_{hn}h+b_{hn}\right)\right) \\
    h' &= (1-z)\odot n+z\odot h
\end{align}
$$

Where

* $x$ is the input tensor of size $\left(N, H_{in}\right)$ or $\left(H_{in}\right)$.
* $h$ is the hidden state tensor of size $\left(N, H_{out}\right)$ or $\left(H_{out}\right)$.
* $W_{ir}$, $W_{iz}$ and $W_{in}$ are weight tensors of size $\left(H_{out}, H_{in}\right)$. 
* $W_{hr}$, $W_{hz}$ and $W_{hn}$ are weight tensors of size $\left(H_{out}, H_{out}\right)$. 
* $\sigma$ is the sigmoid function and can be defined as $\sigma\left(x\right)=\frac{1}{1+e^{-x}}$.
* $\text{tanh}$ is the hyperbolic tangent function and can be defined as $\text{tanh}\left(x\right)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$.
* $\odot$ is the <a href="https://en.wikipedia.org/wiki/Hadamard_product_(matrices)" target="_blank">Hadamard product</a> or element-wise product.
* $b_{ir}$, $b_{iz}$, $b_{in}$, $b_{hr}$, $b_{hz}$ and $b_{hn}$ are bias tensors of size $\left(H_{out}\right)$.

## Complexity
In order to compute the complexity of a single nn.GRUCell, we just need to estimate the number of operations of all four aforementioned equations. For the sake of simplicity, for operations involving sigmoid and hyperbolic tangent, the listed equations will be used and exponentials will be counted as a single operation.

!!! note
    During the following operations, some tensors have to be transposed in order to have compatible dimensions to perform matrix multiplication, even thought this is not explicitly mentioned in <a href="https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html", target="_blank">PyTorch `torch.nn.GRUCel`’s documentation</a>. Additionally, some weight tensors are stacked. For instance, $W_{ir}$, $W_{iz}$ and $W_{in}$ are implemented as a single tensor of size $\left(3\times H_{out}, H_{in}\right)$, and $W_{hr}$, $W_{hz}$ and $W_{hn}$ are implemented as a single tensor of size $\left(3\times H_{out}, H_{out}\right)$, possibly due to efficiency reasons.

### Reset gate
The tensor sizes involved in the operations performed to calculate the reset gate $r$ are

$$
\begin{align}
    r = \sigma\Bigg(\underbrace{\left(H_{out}, H_{in}\right) \times \left(N, H_{in}\right)}_{W_{ir}x} + \underbrace{H_{out}}_{b_{ir}} + \underbrace{\left(H_{out}, H_{out}\right) \times \left(N, H_{out}\right)}_{W_{hr}h} + \underbrace{H_{out}}_{b_{hr}}\Bigg)
\end{align}
$$

In this case, $x$ (with shape $\left(N, H_{in}\right)$) and $h$ (with shape $\left(N, H_{out}\right)$) have to be transposed. Additionally, $b_{ir}$ and $b_{hr}$ will be implicitly broadcasted to be able to be summed with the tensor multiplication results. Then, the unwrapped and transposed shapes involved in the operations are

$$
\begin{align}
    r = \sigma\left(\left(H_{out}, H_{in}\right) \times \left(H_{in}, N\right) + \left(H_{out}, N\right) + \left(H_{out}, H_{out}\right) \times \left(H_{out}, N\right) + \left(H_{out}, N\right)\right)
\end{align}
$$

This will result in a tensor of shape $\left(H_{out}, N\right)$. To estimate the complexity of this operation, it is possible to reuse the results from [`torch.nn.Linear`](./linear.md) for both matrix multiplications and add the sigmoid operations $\sigma$. $r_{ops}$ (the operations of the reset gate $r$) can be then broken down into four parts:

1. The operations needed to compute $W_{ir}x+b_{ir}$.
2. The operations needed to compute $W_{hi}h+b_{hi}$.
3. The operations needed to sum both results.
4. The operations needed to compute the sigmoid function $\sigma$ of this result.

For simplicity sake, the following definitions will be used:

$$
\begin{align}
    r = \sigma\left(\underbrace{W_{ir}x^T+b_{ir}}_{r_0}+\underbrace{W_{hr}h^T+b_{hr}}_{r_1}\right)
\end{align}
$$

Then, in terms of operations ($ops$) when `bias=True`

$$
\begin{align}
    r_{0_{ops}} &=\left(W_{ir}x^T+b_{ir}\right)_{ops} = 2\times N\times H_{out}\times H_{in} \\
    r_{1_{ops}} &=\left(W_{hr}x^T+b_{hr}\right)_{ops} = 2\times N\times H_{out}^2 \\
    \left(r_0+r_1\right)_{ops} &= N\times H_{out} \\
    \sigma_{ops} &= 3\times N\times H_{out} \\
    r_{ops} &= 2\times N\times H_{out}\left(2+H_{in}+ H_{out}\right)
\end{align}
$$

and when `bias=False`

$$
\begin{align}
    r_{0_{ops}} &=\left(W_{ir}x^T+b_{iri}\right)_{ops} = N\times H_{out}\times \left(2\times H_{in}-1\right) \\
    i_{1_{ops}}
    &=\left(W_{hr}x^T+b_{hr}\right)_{ops}=N\times H_{out}\times\left(2\times H_{out}-1\right) \\
    \left(r_0+r_1\right)_{ops} &= N\times H_{out} \\
    \sigma_{ops} &= 3\times N\times H_{out}\\
    r_{ops} &= 2\times N\times H_{out}\times\left(1+H_{in}+ H_{out}\right)
\end{align}
$$

### Update gate
Since the dimensions of this gate are the same as the reset gate $r$, it is trivial to observe that

$$
\begin{equation}
    z_{ops}=r_{ops}
\end{equation}
$$

### $n$
$n$ has a slightly different configuration. Besides the matrix multiplications, there is Hadamard product $\odot$ and an hyperbolic tangent $\text{tanh}$ function. The involved tensor sizes are

$$
\begin{align}
    n = \text{tanh}\Bigg(\underbrace{\left(H_{out}, H_{in}\right) \times \left(N, H_{in}\right)}_{W_{in}x} + \underbrace{H_{out}}_{b_{in}} + \underbrace{\left(H_{out}, N\right)}_{r}\odot\left(\underbrace{\left(H_{out}, H_{out}\right) \times \left(N, H_{out}\right)}_{W_{hn}h} + \underbrace{H_{out}}_{b_{hn}}\right)\Bigg)
\end{align}
$$

Again, it becomes necessary to broadcast and transpose some tensors to be able to perform all operations, resulting in

$$
\begin{align}
    n = \text{tanh}\left(\left(H_{out}, H_{in}\right) \times \left(H_{in}, N\right) + \left(H_{out}, N\right) + \left(H_{out}, N\right)\odot\left(\left(H_{out}, H_{out}\right) \times \left(H_{out}, N\right) + \left(H_{out}, N\right)\right)\right)
\end{align}
$$

Now $n_{ops}$ (the operations performed by $n$) can be divided into five parts:

1. The operations needed to compute $W_{in}x+b_{in}$.
2. The operations needed to compute $W_{hn}h+b_{hn}$.
3. The operations needed to compute the Hadamard product between $r$ and $W_{hn}h+b_{hn}$.
4. The operations needed to sum the terms that result into the $\text{tanh}$ function argument.
5. The operations needed to compute the hyperbolic tangent $\text{tanh}$ function of this result.

Then, the different parts that contribute to $n_{ops}$ can be defined as

$$
\begin{align}
    n = \text{tanh}\left(\underbrace{W_{in}x^T+b_{in}}_{n_0}+r\odot\left(\underbrace{W_{hn}h^T+b_{hn}}_{n_1}\right)\right)
\end{align}
$$

Then, when `bias=True`

$$
\begin{align}
n_{0_{ops}} &= \left(W_{in}x^T+b_{in}\right)_{ops} = 2\times N\times H_{out}\times H_{in} \\
n_{1_{ops}} &= \left(W_{hn}x^T+b_{hn}\right)_{ops} = 2\times N\times H_{out}^2 \\
\left(r\odot n_1\right)_{ops} &= N\times H_{out} \\
\left(n_0+r\odot n_1\right)_{ops} &= N\times H_{out} \\
\text{tanh}_{ops} &= 7\times N\times H_{out} \\ 
n_{ops} &= N\times H_{out}\left(9+2\left(H_{in}+H_{out}\right)\right)
\end{align}
$$

and when `bias=False`

$$
\begin{align}
n_{0_{ops}} &= \left(W_{in}x^T\right)_{ops} = N\times H_{out}\times \left(2\times H_{in}-1\right) \\
n_{1_{ops}} &= \left(W_{hn}x^T\right)_{ops} = N\times H_{out}\times \left(2\times H_{out}-1\right) \\
\left(r\odot n_1\right)_{ops} &= N\times H_{out} \\
\left(n_0+r\odot n_1\right)_{ops} &= N\times H_{out} \\
\text{tanh}_{ops} &= 7\times N\times H_{out} \\ 
n_{ops} &= N\times H_{out}\left(9+2\left(H_{in}+H_{out}-1\right)\right) 
\end{align}
$$

!!! note
    Please note that there are many possible formulations for the amount of operations carried out by $\text{tanh}$. In this calculation, the formula mentioned at the beginning is what is being used. In such a case, there are 7 operations per element are: x4 exponentials, x1 sum, x1 difference and x1 division. Please also note that we are ignoring sign inversion operations, assuming these will usually have a negligible computational cost. 

### $h\prime$
The operations computed to obtain $h\prime$ can be divided into four parts:

1. The operations needed to subtract $\left(1-z\right)$.
2. The operations needed to compute the Hadamard product between $\left(1-z\right)$ and $n$.
3. The operations needed to compute the Hadamard product between $z$ and $h$.
4. The operations needed to sum both Hadamard product results.

In this case, all operations are element wise operations, therefore it is trivial to see that every part will contribute with $N\times H_{out}$ operations, therefore

$$
\begin{align}
h\prime_{ops} &= 4\times N\times H_{out}
\end{align}
$$

### Total complexity
Finally, the total complexity is the sum of all individual contributions

$$
\begin{align}
    \text{GRUCell}_{ops}=r_{ops}+z_{ops}+n_{ops}+h\prime_{ops}
\end{align}
$$

In the case of `bias=True`, the total number of operations is

$$
\begin{align}
    \text{GRUCell}_{ops}&= \underbrace{4\times N\times H_{out}\left(2+H_{in}+ H_{out}\right)}_{r_{ops}+z_{ops}} \nonumber \\
    &\quad + \underbrace{N\times H_{out}\left(9+2\left(H_{in}+H_{out}\right)\right)}_{n_{ops}}
    \nonumber \\
    &\quad + \underbrace{4\times N\times H_{out}}_{h\prime_{ops}}
    \nonumber \\
    \text{GRUCell}_{ops}&= 6\times N \times H_{out} \left(H_{in}+H_{out}+3.5\right)
\end{align}
$$

and for `bias=False`

$$
\begin{align}
    \text{GRUCell}_{ops}&= \underbrace{4\times N\times H_{out}\left(1+H_{in}+ H_{out}\right)}_{r_{ops}+z_{ops}} \nonumber \\
    &\quad + \underbrace{N\times H_{out}\left(9+2\left(H_{in}+H_{out}-1\right)\right)}_{n_{ops}}
    \nonumber \\
    &\quad + \underbrace{4\times N\times H_{out}}_{h\prime_{ops}}
    \nonumber \\
    \text{GRUCell}_{ops}&= 6\times N \times H_{out} \left(H_{in}+H_{out}+2.5\right)
\end{align}
$$

## Summary
The number of operations $\phi$ operformed by a `torch.nn.GRUCell` module can be estimated as

!!! success ""
    === "If `bias=True`"
        $\text{GRUCell}_{ops} = 6\times N \times H_{out} \left(H_{in}+H_{out}+3.5\right)$

    === "If `bias=False`"
        $\text{GRUCell}_{ops} = 6\times N \times H_{out} \left(H_{in}+H_{out}+2.5\right)$

Where

* $N$ is the batch size.
* $H_\text{in}$ is the number of input features.
* $H_\text{out}$ is the number of output features.