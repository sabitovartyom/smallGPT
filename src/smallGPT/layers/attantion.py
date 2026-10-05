import torch
from torch.nn import Module
from .linear import Linear


class CausalSelfAttantion(Module):
    def __init__(self, dim, context_length):
        super().__init__()
        self.dim = dim
        self.context_length = context_length
        self.num_heads = 1

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


class MultiHeadCausalSelfAttantion(Module):
    def __init__(self, dim, context_length, num_heads=4):
        super().__init__()
        if dim < 1 or num_heads < 1 or dim % num_heads != 0:
            raise ValueError("dim must be positive and divisible by num_heads")
        self.dim = dim
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qwery_proj = Linear(dim, dim)
        self.key_proj = Linear(dim, dim)
        self.value_proj = Linear(dim, dim)
        self.output_proj = Linear(dim, dim)

        mask = torch.ones(context_length, context_length, dtype=torch.bool).triu(diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, X):
        without_batch = X.ndim == 2
        if without_batch:
            X = X.unsqueeze(0)
        B, T, D = X.shape
        if T > self.context_length:
            raise ValueError("Sequence length exceeds context_length")
        mask = self.causal_mask[:T, :T]

        qwery = self.qwery_proj(X)
        key = self.key_proj(X)
        value = self.value_proj(X)

        qwery = qwery.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attantion_scores = (qwery @ key.transpose(-2, -1)) / self.head_dim**0.5
        attantion_scores = attantion_scores.masked_fill(mask, float('-inf'))
        attantion_weights = torch.softmax(attantion_scores, dim=-1)
        attantion_output = attantion_weights @ value

        attantion_output = attantion_output.transpose(1, 2).reshape(B, T, D)
        result = self.output_proj(attantion_output)
        if without_batch:
            result = result.squeeze(0)
        return result
