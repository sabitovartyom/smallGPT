import torch
from torch.nn import Module

class GELU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.erf(x / 2**0.5))
