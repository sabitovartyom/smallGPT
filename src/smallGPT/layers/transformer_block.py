from torch.nn import Module
from .layer_norm import LayerNorm
from .attantion import CausalSelfAttantion as Attention
from .feed_forward  import FeedForward

class TransformerBlock(Module):
    def __init__(self, embedding_dim, context_length):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.attention = Attention(embedding_dim, context_length)
        self.feed_forward = FeedForward(embedding_dim, embedding_dim * 4, embedding_dim)
        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

    def forward(self, X):
        X = self.attention(self.norm1(X)) + X
        X = self.feed_forward(self.norm2(X)) + X
        return X
