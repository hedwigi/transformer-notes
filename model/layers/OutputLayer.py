import torch.nn as nn
import torch.nn.functional as F


class OutputLayer(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(OutputLayer, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return F.log_softmax(self.proj(x), dim=-1)
