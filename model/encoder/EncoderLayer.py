import torch.nn as nn
from util.ModelUtil import ModelUtil
from model.common.SublayerConnection import SublayerConnection


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = ModelUtil.clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
            - self_attn, residual & norm
            - feed_forward, residual & norm

        :param x:
        :param mask:
        :return:
        """
        att_op = lambda x: self.self_attn(x, x, x, mask)
        x = self.sublayers[0](x, att_op)
        return self.sublayers[1](x, self.feed_forward)
