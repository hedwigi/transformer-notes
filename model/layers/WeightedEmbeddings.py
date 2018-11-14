import math
import torch.nn as nn


class WeightedEmbeddings(nn.Module):
    """
        Lookup table, multiplied by sqrt(d_model)
    """

    def __init__(self, d_model, vocab):
        """

        :param d_model: the size of each embedding vector
        :param vocab: size of the dictionary of embeddings
        """
        super(WeightedEmbeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """

        :param x: [batch_size, max_len]
        :return: [batch_size, max_len, embedding_dim)
        """
        return self.lut(x) * math.sqrt(self.d_model)
