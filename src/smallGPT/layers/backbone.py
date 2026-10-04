from torch.nn import Module

from .embedding import Embedding
from .positional_embedding import PositionalEmbedding
from .transformer_block import TransformerBlock
from .layer_norm import LayerNorm
from .linear import Linear


class Backbone(Module):
    def __init__(self, vocab_size, embedding_dim, context_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(context_length, embedding_dim)
        self.transformer_block1 = TransformerBlock(embedding_dim, context_length)
        self.transformer_block2 = TransformerBlock(embedding_dim, context_length)
        self.final_norm = LayerNorm(embedding_dim)
        # LM head: для каждой позиции получить оценку каждого токена словаря.
        self.linear = Linear(embedding_dim, vocab_size, biasFlag=False)

    def forward(self, X):
        X = self.embedding(X)
        X = self.positional_embedding(X)
        X = self.transformer_block1(X)
        X = self.transformer_block2(X)
        X = self.final_norm(X)
        return self.linear(X)
