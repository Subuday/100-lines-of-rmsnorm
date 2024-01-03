import torch
from torch import nn


class RSMNorm(nn.Module):

    def __init__(self, d):
        super(RSMNorm, self).__init__()
        self.d = d
        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

    def forward(self, x):
        x_norm = x.norm(2, dim=-1, keepdim=True)
        rsm_norm = x_norm * self.d ** (-1. / 2)
        return x / rsm_norm * self.scale


if __name__ == "__main__":
    t = torch.randn(10, 3)
    rsm = RSMNorm(3)
    rsm(t)
