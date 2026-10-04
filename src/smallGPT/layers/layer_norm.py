import torch
from torch.nn import Module, Parameter

class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.gamma = Parameter(torch.ones(dim))
        self.beta = Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        return self.gamma * (x - mean) / (var + self.eps)**0.5 + self.beta