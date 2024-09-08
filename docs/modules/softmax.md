# Softmax (`torch.nn.Softmax`)
A `torch.nn.Softmax` applies the softmax function along a given dimension `dim`.
This function is defined as

$$
\begin{align}
    \text{Softmax}\left(x_i\right) = \frac{e^{x_i}}{\sum_j e^{x_j}}
\end{align}
$$

Where $x$ is the input tensor of any size because $\text{Softmax}$ is an element-wise activation function.

!!! note
    This function causes the sum of all values along `dim` to be `1.0`. This function is normally used for feature selection and multi-class classification, producing values that can be interpreted as weights for a certain feature, or the probability of a certain outcome. However, it is important to consider in the latter case that such values are not inherently calibrated probabilities, unless the network is explicitly trained for this purpose.


## Complexity
The denominator of the computation $\sum_je^{x_j}$ needs to be calculated only once per row along `dim`. This calculation involves as many exponential functions as elements along `dim` and then a sum. If we assume that `dim` as $N$ elements, then the number of operations is

$$
\begin{equation}
\left(\sum_je^{x_j}\right)_{ops}=2\times N - 1
\end{equation}
$$

Where $N$ is the number of exponential function operations, and $N-1$ is the number of additions. Then, for the numerator there is a per-element exponential ($N$ exponential operations) and a per element division ($N$ division operations), totalling $2 \times N$  additional operations resulting in $4\times N - 1$ operations per row.
This amount of operations will be repeated $M$ times where $M$ corresponds to the dimensions other than dim

$$
\begin{equation}
    M=\prod_{n\neq\text{dim}}^{N - 1}d_n=d_0\times d_1\times\cdots\times d_{N-1}
\end{equation}
$$

Now, the resulting complexity is
$$
\begin{equation}
    \left(\text{Softmax}\right)_{ops}=M\times\left(4\times N - 1\right)
\end{equation}
$$

## Summary
The number of operations performed by a `torch.nn.Softmax` module can be estimated as

!!! success ""
    $\left(\text{Softmax}\right)_{ops}=M\times\left(4\times N - 1\right)$

Where

* $N$ is the number of elements along dimension `dim`.
* $M$ is the product of the size of all other dimensions except `dim`.