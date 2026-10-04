import torch
from torch.nn import Module, Parameter

class Linear(Module):
    def __init__(self, input_dim, output_dim, biasFlag=True):
        super().__init__()
        self.biasFlag = biasFlag
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.empty(output_dim, input_dim))
        self.bias = Parameter(torch.empty(output_dim)) if biasFlag else None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.weight.uniform_(-1 / self.input_dim**0.5, 1 / self.input_dim**0.5)
        if self.bias is not None:
            self.bias.zero_()

    def forward(self, X):
        if self.bias is None:
            y = X @ self.weight.T
        else :
            y = X @ self.weight.T + self.bias
        return y    