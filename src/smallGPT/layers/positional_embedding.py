import torch
from torch.nn import Module
from .embedding import Embedding


class PositionalEmbedding(Module):
    def __init__(self, context_length, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.embedding = Embedding(context_length, embedding_dim)

    def forward(self, X):
        T = X.shape[-2]
        if T > self.context_length:
            raise ValueError("Sequence length exceeds context_length")
        return X + self.embedding(torch.arange(T, device=X.device))
