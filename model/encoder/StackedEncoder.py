import torch.nn as nn
from model.common.LayerNorm import LayerNorm
from util.ModelUtil import ModelUtil


class StackedEncoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(StackedEncoder, self).__init__()
        self.layers = ModelUtil.clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)