# Linear (`torch.nn.Linear`)
A linear layer computes the following operation in a forward pass:

$$
\begin{equation}y=xA^T+b\end{equation}
$$

Where

* $x$ is a rank-N tensor of size $\left(\ast, H_\text{in}\right)$ with N $\geq$ 1
* $A$ is a weight rank-2 tensor of size  $\left(H_\text{out}, H_\text{in}\right)$.
* $b$ is a bias rank-1 tensor of size $\left(H_\text{out}\right)$.
* $y$ is the output rank-N tensor of size $\left(\ast, H_\text{out}\right)$.
* $\ast$ means any number of dimensions.
* $H_\text{in}$ is the number of input features.
* $H_\text{out}$ is the number of output features.

The weight tensor $A$ will apply a linear transformation or mapping to the input tensor $x$, whereas the bias tensor $b$ can be though of as a DC offset, since it is a learnable term that will act as a constant that is added to the result of the tensor-tensor multiplication $xA^T$. 

## Complexity
A linear module involves two tensor-tensor operations: one multiplication and one addition. In order to simplify the calculations, 
$x$ will be assumed to have size $\left(1, H_\text{in}\right)$. After computing the results, they will be expanded for higher dimensions. If $A$ is a rank-2 tensor of size $\left(H_\text{out}, H_\text{in}\right)$, then $A^T$ has size $\left(H_\text{in}, H_\text{out}\right)$. Therefore

$$
\begin{equation}
xA^T=\begin{pmatrix} x_0 & ...&x_{H_\text{in}-1}
\end{pmatrix}\times\begin{pmatrix} a_{0,0} & \cdots&a_{0,H_\text{out}-1}\\ \vdots &  \ddots & \vdots \\ a_{H_\text{in}-1,0}& \cdots & a_{H_\text{in}-1,H_\text{out}-1}
\end{pmatrix}
\end{equation}
$$

A single element $y_n$ of the output tensor corresponds to the dot product of the first (and only) row of tensor $x$ and a column of $A^T$. As an example, the first output element $y_0$ will be computed as

$$
\begin{equation}
y_0=\sum\limits_{n=0}^{H_\text{in}-1}x_n a_{n, 0}=x_0 a_{0, 0}+x_1 a_{1, 0}+\cdots+x_{H_\text{in}-1} a_{H_\text{in}-1,0}
\end{equation}
$$

This operation requires $H_\text{in}$ multiplications and $H_\text{in} - 1$ additions. Therefore, the total number of operations per output feature is $2 \times H_\text{in} - 1$. This has to be repeated $H_\text{out}$ times. Then, the total number of operations $\phi$ so far is

$$
\begin{equation}
\phi=H_\text{out}\times\left(2 \times H_\text{in} - 1\right)
\end{equation}
$$

Next, it is necessary to add the bias tensor $b$. This is rather straightforward, since the result of $xA^T$ has shape $\left(1, H_\text{out}\right)$ and the bias tensor $b$ has shape $\left(H_\text{out}\right)$. The addition of the bias term corresponds therefore to $H_\text{out}$ additions

$$
\begin{equation}
\phi=H_\text{out}\times\left(2 \times H_\text{in} - 1\right) + H_\text{out} = 2\times H_\text{in}\times H_\text{out}
\end{equation}
$$

Depending on whether module was instantiated using `bias=True` or `bias=False`, there are two possible outcomes

$$
\begin{equation}
\phi=\begin{cases}
    2\times H_\text{in}\times H_\text{out}, & \text{if}\ \text{bias}=\text{True} \\
    H_\text{out}\times\left(2 \times H_\text{in} - 1\right), &\text{if}\ \text{bias}=\text{False}
\end{cases}
\end{equation}
$$

Finally, it is necessary to add the batch size. Since $\ast$ is any set of dimensions of $x$. Given a rank-N tensor $x$ of size $\left(d_0, d_1, \cdots, d_{N-1}\right)$ it is possible to define the batch size $\beta$ as

$$
\begin{equation}
\beta=\prod^{N - 2}d_n=d_0\times d_1\times\cdots\times d_{N-2}
\end{equation}
$$

!!! note
    Please note that `torch.nn.Linear` allows the batch size $\beta$ to be composed of a single dimension or many, so its definition slightly differs from the batch size definition of other type of modules. As an example, if the input tensor $x$ has size $\left(2, 3, 4\right)$ then the batch dimension is $6$, and the number of input features $H_\text{in}$ is $4$. This is because `torch.nn.Linear` considers only the very last dimension as input features.

The previously calculated number of operations is then repeated $\beta$ times. Finally, the total number of operations per forward pass is

$$
\begin{equation}
\phi=\begin{cases}
    2\times\beta\times H_\text{out}\times H_\text{in}, & \text{if}\ \text{bias}=\text{True} \\
    \beta\times H_\text{out}\times\left(2 \times H_\text{in} - 1\right), &\text{if}\ \text{bias}=\text{False}
\end{cases}
\end{equation}
$$

## Summary
The number of operations $\phi$ performed by a `torch.nn.Linear` module can be estimated as

!!! success ""

    === "If `bias=True`"
        $\phi = 2\times\beta\times H_\text{out}\times H_\text{in}$

    === "If `bias=False`"
        $\phi = \beta\times H_\text{out}\times\left(2 \times H_\text{in} - 1\right)$

Where

* $H_\text{in}$ is the number of input features.
* $H_\text{out}$ is the number of output features.
* $\beta$ is the batch size. For the case of `torch.nn.Linear` and a rank-N input tensor $x$ of size $\left(d_0, d_1, \cdots, d_{N-1}\right)$ it is defined as $d_0\times d_1\times\cdots\times d_{N-2}$. 