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
* $\epsilon$ is the machine epsilon added to avoid dividing by zero.
* $\gamma$ and $\beta$ are learnable parameters that are present if `elementwise_affine=True`.

!!! note
    The standard deviation is calculated using a biased estimator, which is equivalent to `torch.var(input, correction=0)`.


## Complexity
The complexity of a `torch.nn.LayerNorm` layer can be divided into two parts: The aggregated statistics calculation (i.e. mean and standard deviation) and the affine transformation applied by $\gamma$ and $\beta$ if `elementwise_affine=True`.

### Aggregated statistics
The complexity of the mean corresponds to the sum of all elements in the last $D$ dimensions of the input tensor $x$ and the division of that number by the total number of elements. As an example, if `normalized_shape=(3, 5)` then there are 14 additions and 1 division. This also corresponds to the product of the dimensions involved in `normalized_shape`.

$$
\begin{equation}
    \left(\text{E}\left[x\right]\right)_{ops} = \prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]
\end{equation}
$$

Once $\text{E}\left[x\right]$ is obtained, it can be reused to obtain the variance using <a href="https://pytorch.org/docs/stable/generated/torch.var.html" target="blank">`torch.var`</a> that is defined as

$$
\begin{equation}
    \text{Var}\left[x\right] = \frac{1}{\text{max}\left(0, N-\delta N\right)}\sum_{i=0}^{N-1}\left(x_i-\text{E}\left[x\right]\right)
\end{equation}
$$

Where $\delta N$ is the correction (0 in this case). This step involves an element-wise subtraction, $N-1$ additions to compute the sum. Additionally, a subtraction, a $\text{max}$ operation and a division are necessary to resolve the fraction. Then

$$
\begin{equation}
    \left(\text{Var}\left[x\right]\right)_{ops} = 2+2\times\prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]
\end{equation}
$$

Now, there are 2 additional operations (an addition and a square root) to obtain $\sqrt{\text{Var}\left[x\right]+\epsilon}$, therefore

$$
\begin{equation}
    \left(\sqrt{\text{Var}\left[x\right]+\epsilon}\right)_{ops} = 4+2\times\prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]
\end{equation}
$$

Finally, to obtain the whole fraction there is an additional element-wise subtraction in the numerator, and an element-wise division to divide the numerator by the denominator, therefore

$$
\begin{equation}
    \left(\frac{x-\text{E}\left[x\right]}{\sqrt{\text{Var}\left[x\right]+\epsilon}}\right)_{ops} = 4+5\times\prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]
\end{equation}
$$

### Elementwise affine
If `elementwise_affine=True`, there is an element-wise multiplication by $\gamma$. If `bias=True`, there is also an element-wise addition by $\beta$. Therefore the whole complexity of affine transformations is

$$
\begin{equation}
    \gamma_{ops} = \prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]
\end{equation}
$$

when `bias=False`, and

$$
\begin{equation}
    \gamma_{ops}+\beta_{ops} = 2\times\prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]
\end{equation}
$$

when `bias=True`.

### Batch size
So far we have not included the batch size $N$, which in this case could be defined as all other dimensions that are not $D$. This means, those that are not included in `normalized_shape`.

!!! note
    Please note that $N$ here corresponds to all dimensions not included in `normalized_shape`, which is different from the definition ot $N$ in `torch.var` which corresponds to the number of elements in the input tensor of that function.  

The batch size $N$ multiplies all previously calculated operations by a factor $\eta$ corresponding to the multiplication of the remaining dimensions. For example, if the input tensor has size `(2, 3, 5)` and `normalized_shape=(3, 5)`, then $\eta$ is $2$.

### Total complexity
Including all previously calculated factor, the total complexity can be summarized as

$$
\begin{equation}
    \text{LayerNorm}_{ops} = \eta\left(4+5\times\prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]\right)
\end{equation}
$$

if `elementwise_affine=False` or 

$$
\begin{equation}
    \text{LayerNorm}_{ops} = \eta\left(4+6\times\prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]\right)
\end{equation}
$$

if `elementwise_affine=True` and `bias=False`, and

$$
\begin{equation}
    \text{LayerNorm}_{ops} = \eta\left(4+7\times\prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]\right)
\end{equation}
$$

if `elementwise_affine=True` and `bias=True`

## Summary
The number of operations performed by a `torch.nn.LayerNorm` module can be estimated as

!!! success ""
    === "If `elementwise_affine=False`"
        $\text{LayerNorm}_{ops} = \displaystyle\eta\left(4+5\times\prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]\right)$ 
    
    === "If `elementwise_affine=True` and `bias=False`"
        $\text{LayerNorm}_{ops} = \displaystyle\eta\left(4+6\times\prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]\right)$
    
    === "If `elementwise_affine=True` and `bias=True`"
        $\text{LayerNorm}_{ops} = \displaystyle\eta\left(4+7\times\prod_{d=0}^{D-1}\text{normalized\_shape}[\text{d}]\right)$
    
Where

* $\eta$ is the multiplication of all dimensions that are not included in `normalized_shape`.
* $D$ is number of the last dimensions included in `normalized_shape`.

As an example, if the input tensor has size `(2, 3, 5)` and `normalized_shape=(3, 5)`, then $D=15$ and $\eta=2$.
