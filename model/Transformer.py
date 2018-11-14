import torch.nn as nn


class Transformer(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_input_embed_op, tgt_input_embed_op, output_layer):
        """

        :param encoder:
        :param decoder:
        :param src_input_embed_op: x and positional emb, return shape [batch_size, max_len, d_model]
        :param tgt_input_embed_op: [batch_size, max_len, d_model]
        :param output_layer:
        """
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_input_embed_op = src_input_embed_op
        self.tgt_input_embed_op = tgt_input_embed_op
        self.output_layer = output_layer

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
            Take in and process masked src and target sequences.
            - encode operations
                * src_input_embed_op
                * encoder operations
            - decode operations
                * tgt_input_embed_op
                * decoder operations

        :param src:
        :param tgt:
        :param src_mask:
        :param tgt_mask:
        :return:
        """
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
            - src_input_embed_op
            - encoder operations
        :param src:
        :param src_mask:
        :return:
        """
        return self.encoder(self.src_input_embed_op(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
            - tgt_input_embed_op
            - decoder operations
        :param memory:
        :param src_mask:
        :param tgt:
        :param tgt_mask:
        :return:
        """
        return self.decoder(self.tgt_input_embed_op(tgt), memory, src_mask, tgt_mask)
