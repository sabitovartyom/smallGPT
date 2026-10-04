from torch.nn import Module
from .linear import Linear
from .gelu import GELU

class FeedForward(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.linear1 = Linear(input_dim, hidden_dim)
        self.gelu = GELU()
        self.linear2 = Linear(hidden_dim, output_dim)

    def forward(self, X):
        X = self.linear1(X)
        X = self.gelu(X)
        X = self.linear2(X)
        return X