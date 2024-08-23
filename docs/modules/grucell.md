# GRUCell (`torch.nn.GRUCell`)
A `torch.nn.GRUCell` corresponds to a single cell of a Grated Recurrent Unit (`torch.nn.GRU`). A `torch.nn.GRUCell` takes an **input** $x$, a **hidden state** $h$. Internally, it
has a **reset gate** $r$ and an **update gate** $z$ that help to propagate information between time steps. These are combined to generate $n$, that is then used to create a new hidden state $h\prime$. The relationship between these tensors is defines as

$$
\begin{align}
    \tag*{(1)} r &= \sigma\left(W_{ir}x+b_{ir}+W_{hr}h+b_{hr}\right) \\
    \tag*{(2)} z &= \sigma\left(W_{iz}x+b_{iz}+W_{hz}h+b_{hz}\right) \\
    \tag*{(3)} n &= \text{tanh}\left(W_{in}x+b_{in}+r\odot\left(W_{hn}h+b_{hn}\right)\right) \\
    \tag*{(4)} h' &= (1-z)\odot n+z\odot h
\end{align}
$$

Where