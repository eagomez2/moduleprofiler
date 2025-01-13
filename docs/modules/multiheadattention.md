# MultiheadAttention (`torch.nn.MultiheadAttention`)
A `torch.nn.MultiheadAttention` allows attending specific elements of the input. In practice, it means producing a weighting mask to assign different importance to different elements in a sequence in a given task. It was popularized by the paper <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a> by Vaswani et al. in 2017. A `torch.nn.MultiheadAttention` layer can be described through

$$
\begin{align}
\text{MultiheadAttention}\left(Q, K, V\right)&=\text{Concat}\left(\text{head}_0, \text{head}_1, ..., \text{head}_h\right)W^{o}
\end{align}
$$

Where each head computed

$$
\begin{align}
\text{head}_i&=\text{Attention}\left(Q, K, V\right)&= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\end{align}
$$

Where

* $Q$ is known ad the query tensor and is used to calculate the attention scores, which determine how much focus should be given to different elements.
* $K$ is the key tensor that is compared with the query to find relevant information.
* $V$ is the value tensor that contains the actual data that is weighted and used in the output.

!!! note
    The $Q$, $K$ and $V$ tensors do not have fixed dimensions since they correspond to inputs of the `forward()` method of an `nn.MultiheadAttention` instance. When $Q$, $K$ and $V$ are the same tensors, this layer is also known and **self-attention**. In addition, an **attention mask** can be added to the $\text{softmax}$ argument in such a way that certain elements will be ignored in the attention tensor that is multiplied by $V$. This is typically used to design causal mechanisms in which each element can only pay attention to elements with the same or previous indices in the sequence. 

## Complexity

## Summary