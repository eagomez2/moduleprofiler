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

## Aggregaed statistics


## Elementwise affine

## Summary