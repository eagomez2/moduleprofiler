# Sigmoid (`torch.nn.Sigmoid`)
A `torch.nn.Sigmoid` corresponds to a sigmoid function that can be
defined as

$$
\begin{align}
    \text{Sigmoid}\left(x\right) = \sigma\left(x\right) = \frac{1}{1+e^{-x}}
\end{align}
$$

Where $x$ is the input tensor of any size because $\text{Sigmoid}$ is an element-wise activation function.

## Complexity
For this case, there one exponential, one sum and one division. Therefore, the total complexity is simply three times the number of elements in the input tensor $x$. Given a rank-N tensor $x$ of size $\left(d_0, d_1, \cdots, d_{N-1}\right)$ the number of operations performed by a `torch.nn.Sigmoid` module $\text{Sigmoid}_{ops}$ is

$$
\begin{equation}
    \text{Sigmoid}_{ops}=3\times\prod^{N - 1}d_n=d_0\times d_1\times\cdots\times d_{N-1}
\end{equation}
$$

!!! note
    Please note that calculating an exponential is generally much more complex than performing a single multiplication. However, since the specific implementation details are not covered by this package, we assume an increase of one operation for both cases.

## Summary
The number of operations performed by a `torch.nn.Sigmoid` module can be estimated as

!!! success ""
    $\text{Sigmoid}_{ops}=3\times\prod^{N - 1}d_n=d_0\times d_1\times\cdots\times d_{N-1}$

Where $x$ is a rank-N tensor of size $\left(d_0, d_1, \cdots, d_{N-1}\right)$.