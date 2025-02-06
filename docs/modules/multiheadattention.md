# MultiheadAttention (`torch.nn.MultiheadAttention`)
A `torch.nn.MultiheadAttention` allows attending specific elements of the input. In practice, it means producing a weighting mask to assign different importance to different elements in a sequence in a given task. It was popularized by the paper <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a> by Vaswani et al. in 2017. A `torch.nn.MultiheadAttention` layer can be described through

$$
\begin{align}
\text{MultiheadAttention}\left(Q, K, V\right)&=\text{Concat}\left(\text{head}_0, \text{head}_1, ..., \text{head}_h\right)W^{o}
\end{align}
$$

Where each head computes

$$
\begin{align}
\text{head}_i&=\text{Attention}\left(Q, K, V\right)&= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\end{align}
$$

Where

* $Q$ is known as the query tensor and is used to calculate the attention scores, which determine how much focus should be given to different elements.
* $K$ is the key tensor that is compared with the query to find relevant information.
* $V$ is the value tensor that contains the actual data that is weighted and used in the output.

!!! note
    The $Q$, $K$ and $V$ tensors do not have fixed dimensions since they correspondto inputs of the `forward()` method of an `nn.MultiheadAttention` instance. When $Q$, $K$ and $V$ are the same tensors, this layer is also known and **self-attention**. In addition, an **attention mask** can be added to the $\text{softmax}$ argument in such a way that certain elements will be ignored in the attention tensor that is multiplied by $V$. This is typically used to design causal mechanisms in which each element can only pay attention to elements with the same or previous indices in the sequence. 

## Complexity
A multihead attention module involves one tensor-tensor multiplication ($QK^T$), an element-wise division by a factor
of $\sqrt{d_k}$, a softmax operation and a tensor-tensor multiplication where the resulting factor is multiplied by the tensor $V$.
However, the query $Q$, key $K$ and value $V$ tensors corresponding to weighted versions of the layer inputs by their respective weights $W^Q$, $W^K$ and $W^V$, therefore

$$
\begin{align}
Q=Q_{in}W^Q \\
K=K_{in}W^K \\ 
V=V_{in}W^V
\end{align}
$$

Where

* $Q_{in}$ is a tensor of shape $\left(L, E_{q}\right)$ and $W^Q$ is a tensor of shape $\left(E_{q}, E_{q}\right)$.
* $K_{in}$ is a tensor of shape $\left(L, E_{k}\right)$ and $W^K$ is a tensor of shape $\left(E_{k}, E_{k}\right)$.
* $V_{in}$ is a tensor of shape $\left(L, E_{v}\right)$ and $W^V$ is a tensor of shape $\left(E_{v}, E_{v}\right)$.
* $L$ is the sequence length and $E{q}$, $E_{k}$ and $E_{v}$ are the embedding dimensions.

!!! note
    Please notice that we are currently ignoring the batch size because it will be added later on in our calculations. Additionally, the specified dimensions assume all tensor-tensor multiplications are compatible.

In terms of complexity

$$
\begin{align}
Q_{ops}=L\times E_{q}\times\left(2\times E_{q}-1\right) \\
K_{ops}=L\times E_{k}\times\left(2\times E_{k}-1\right) \\ 
V_{ops}=L\times E_{v}\times\left(2\times E_{v}-1\right)
\end{align}
$$

Then 

$$
\begin{align}
\left(\frac{QK^T}{\sqrt{d_k}}\right)_{ops} &= L^2\times\left(2\times E_k\right)+L\times E_k
\end{align}
$$

## Summary
