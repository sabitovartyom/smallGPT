from torch.nn import Module

from .embedding import Embedding
from .positional_embedding import PositionalEmbedding
from .transformer_block import TransformerBlock
from .layer_norm import LayerNorm
from .linear import Linear


class Backbone(Module):
    def __init__(self, vocab_size, embedding_dim, context_length, num_layers=4, num_heads=4):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(context_length, embedding_dim)
        for index in range(1, num_layers + 1):
            setattr(self, f"transformer_block{index}", TransformerBlock(embedding_dim, context_length, num_heads))
        self.final_norm = LayerNorm(embedding_dim)
        self.linear = Linear(embedding_dim, vocab_size, biasFlag=False)

    def forward(self, X):
        X = self.embedding(X)
        X = self.positional_embedding(X)
        for index in range(1, self.num_layers + 1):
            X = getattr(self, f"transformer_block{index}")(X)
        X = self.final_norm(X)
        return self.linear(X)
