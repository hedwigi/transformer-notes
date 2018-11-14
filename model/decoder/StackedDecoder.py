import torch.nn as nn
from util.ModelUtil import ModelUtil
from model.common.LayerNorm import LayerNorm


class StackedDecoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(StackedDecoder, self).__init__()
        self.layers = ModelUtil.clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
