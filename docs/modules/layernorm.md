# LayerNorm (`torch.nn.LayerNorm`)
A `torch.nn.LayerNorm` module computes the mean and standard deviation over the last $D$ dimensions specified by the `normalized_shape` parameter. If `elementwise_affine=True`, then two learnable parameters $\gamma$ and $\beta$ apply also an element-wise affine transformation that can be described as

$$
\begin{equation}
    y=\frac{x-\text{E}\left[x\right]}{\sqrt{\text{Var}\left[x\right]+\epsilon}}\times \gamma + \beta
\end{equation}
$$

Where

* $x$ is the input of size $\left(N, \ast\right)$
* $\text{E}\left[x\right]$ is the mean of $x$ over the last $D$ dimensions.
* $\text{Var}\left[x\right]$ is the variance of $x$ over the last $D$ dimensions.
* $\epsilon$ is the machine epsilon added to avoid divisions by zero.
* $\gamma$ and $\beta$ are learnable parameters present if `elementwise_affine=True`.

!!! note
    The standard deviation is calculated using a biased estimator, which is equivalent to `torch.var(input, correction=0)`.


## Complexity
The complexity of a `torch.nn.LayerNorm` layer can be divided into two parts: The aggregated statistics calculation (i.e. mean and standard deviation) and the affine transformation applied by $\gamma$ and $\beta$ if `elementwise_affine=True`.

## Aggregated statistics
The complexity of the mean corresponds to the sum of all elements in the last $D$ dimensions of the input tensor $x$ and the division of that number by the total number of elements. As an example, if `normalized_shape=(3, 5)` then there are 14 additions and 1 division. This also corresponds to the product of the dimensions involved in `normalized_shape`.

$$
\begin{equation}
    \left(\text{E}\left[x\right]\right)_{ops} = \prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]
\end{equation}
$$

Once $\text{E}\left[x\right]$ is obtained, it can be reused to obtain the variance using <a href="https://pytorch.org/docs/stable/generated/torch.var.html" target="blank">`torch.var`</a> that for `correction=0` reduces to

$$
\begin{equation}
    \text{Var}\left[x\right] = \frac{1}{\text{max}\left(0, N\right)}\sum_{i=0}^{N-1}\left(x_i-\text{E}\left[x\right]\right)
\end{equation}
$$

This step involves an element-wise subtraction, $N-1$ additions, a division and a $\text{max}$ operation. Then

$$
\begin{equation}
    \left(\text{Var}\left[x\right]\right)_{ops} = 1+2\times\prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]
\end{equation}
$$

Now, there are 2 additional operations (an addition and a square root) to obtain $\sqrt{\text{Var}\left[x\right]+\epsilon}$, therefore

$$
\begin{equation}
    \left(\sqrt{\text{Var}\left[x\right]+\epsilon}\right)_{ops} = 2+2\times\prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]
\end{equation}
$$

Finally, to obtain the whole fraction there is an additional element-wise subtraction in the numerator, and an element-wise division to divide the numerator by the denominator, therefore

$$
\begin{equation}
    \left(\frac{x-\text{E}\left[x\right]}{\sqrt{\text{Var}\left[x\right]+\epsilon}}\right)_{ops} = 3+5\times\prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]
\end{equation}
$$

## Elementwise affine

## Summary