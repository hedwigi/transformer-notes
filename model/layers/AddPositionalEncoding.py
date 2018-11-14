import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class AddPositionalEncoding(nn.Module):
    """
        positional embeddings are same to any words with the same position and d_model
        initialized only once
    """
    def __init__(self, d_model, dropout, max_len=5000):
        """

        :param d_model:
        :param dropout:
        :param max_len:
        """
        super(AddPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)                                  # [max_len,]
        position = torch.arange(0, max_len).unsqueeze(1)                    # [max_len, 1]

        # NOTES: this implementation only allow even d_model
        # (if d_model = 3 or 5.., div_term will be different for sin/cos)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            dropout(sentence embedding + positional embedding)
        :param x: Variable, input embedding of sentences with [1, max_len, d_model]
        :return: Variable, [1, max_len, d_model]
        """
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


if __name__ == "__main__":

    d_model = 4
    max_len = 5

    pe = AddPositionalEncoding(d_model, 0, max_len)

    # [max_len, d_model]
    sent_emb = Variable(torch.FloatTensor(
                        [[0.2, 4.2, 1.3, 2.9],
                       [3.2, 2.8, 3.6, 1.5],
                       [1.2, 3.2, 2.4, 3.2],
                       [2.3, 1.8, 3.2, 4.6],
                       [2.5, 2.9, 4.0, 0.2]]))
    # [1, max_len, d_model]
    sent_emb = sent_emb.unsqueeze(0)

    # sent_emb + position_emb
    y = pe(sent_emb)

    print(y)
