import torch
from torch.nn import Module
from .linear import Linear


class CausalSelfAttantion(Module):
    def __init__(self, dim, context_length):
        super().__init__()
        self.dim = dim
        self.context_length = context_length

        self.qwery_proj = Linear(dim, dim)
        self.key_proj = Linear(dim, dim)
        self.value_proj = Linear(dim, dim)
        self.output_proj = Linear(dim, dim)

        mask = torch.ones(context_length, context_length, dtype=torch.bool).triu(diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, X):
        T = X.shape[-2]
        if T > self.context_length:
            raise ValueError("Sequence length exceeds context_length")
        mask = self.causal_mask[:T, :T]

        qwery = self.qwery_proj(X)
        key = self.key_proj(X)
        value = self.value_proj(X)
        attantion_scores = (qwery @ key.transpose(-2, -1)) / self.dim**0.5
        attantion_scores = attantion_scores.masked_fill(mask, float('-inf'))
        attantion_weights = torch.softmax(attantion_scores, dim=-1)
        attantion_output = attantion_weights @ value
        return self.output_proj(attantion_output)
