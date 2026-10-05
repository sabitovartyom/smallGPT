from torch.nn import Module
from .layer_norm import LayerNorm
from .attantion import CausalSelfAttantion, MultiHeadCausalSelfAttantion
from .feed_forward  import FeedForward

class TransformerBlock(Module):
    def __init__(self, embedding_dim, context_length, num_heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        if num_heads == 1:
            self.attention = CausalSelfAttantion(embedding_dim, context_length)
        else:
            self.attention = MultiHeadCausalSelfAttantion(embedding_dim, context_length, num_heads)
        self.feed_forward = FeedForward(embedding_dim, embedding_dim * 4, embedding_dim)
        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

    def forward(self, X):
        X = self.attention(self.norm1(X)) + X
        X = self.feed_forward(self.norm2(X)) + X
        return X
