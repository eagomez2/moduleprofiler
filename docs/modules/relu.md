# ReLU (`torch.nn.ReLU)`
A `torch.nn.ReLU` corresponds to a Rectified Linear Unit function that can be
defined as

$$
\begin{align}
    \text{ReLU}\left(x\right) = \text{max}\left(0, x\right)
\end{align}
$$

Where $x$ is the input tensor of any size because $\text{ReLU}$ is an element-wise activation function.

## Complexity
For this case, one operation per element is assumed, therefore the total complexity is simply the number of elements of $x$. Given a rank-N tensor $x$ of size $\left(d_0, d_1, \cdots, d_{N-1}\right)$ the number of operations performed by a `torch.nn.ReLU` module $\text{ReLU}_{ops}$ is

$$
\begin{equation}
\text{ReLU}_{ops}=\prod^{N - 1}d_n=d_0\times d_1\times\cdots\times d_{N-1}
\end{equation}
$$

!!! note
    Please note `max` is not a basic arithmetic operation and the actual number of instructions this function requires may vary. Since `moduleprofiler` is based
    on the mathematical relationship between input and output, one operation per
    element is assumed for this activation function. 

## Summary
The number of operations performed by a `torch.nn.ReLU` module can be estimated as

!!! success ""
    $\text{ReLU}_{ops}=\prod^{N - 1}d_n=d_0\times d_1\times\cdots\times d_{N-1}$

Where $x$ is a rank-N tensor of size $\left(d_0, d_1, \cdots, d_{N-1}\right)$. 
