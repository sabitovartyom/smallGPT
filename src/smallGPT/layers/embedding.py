import torch
from torch.nn import Module, Parameter

class Embedings(Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.weight = Parameter(torch.empty(vocab_size, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
            torch.nn.init.normal_(self.weight, mean=0.0, std=0.2)

    def forward(self, token_ids):
        return self.weight[token_ids]

