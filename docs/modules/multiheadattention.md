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

In this operation, $Q$ is known ad the query, $K$ as the key, and $V$ as the value tensor, respectively. The query $Q$ is used to calculate attention scores, which determine how much focus should be given to different elements. The key $K$ is compared with the query to find relevant information, and the value $V$ contains the actual data that is weighted and used in the output.