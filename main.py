import numpy as np
import torch
import torch.nn as nn
import copy, time
import pickle
import os
from torch.autograd import Variable
from model.layers.MultiHeadedAttention import MultiHeadedAttention
from model.layers.PositionwiseFeedForward import PositionwiseFeedForward
from model.layers.AddPositionalEncoding import AddPositionalEncoding
from model.Transformer import Transformer
from model.encoder.StackedEncoder import StackedEncoder
from model.encoder.EncoderLayer import EncoderLayer
from model.decoder.StackedDecoder import StackedDecoder
from model.decoder.DecoderLayer import DecoderLayer

from model.layers.WeightedEmbeddings import WeightedEmbeddings
from model.layers.OutputLayer import OutputLayer

from model.regularization.LabelSmoothing import LabelSmoothing
from model.optimizer.NoamOpt import NoamOpt
from train_helper.SimpleLossCompute import SimpleLossCompute
from data_helper.Batch import Batch
from util.DataUtil import DataUtil


def make_model(src_vocab_size, tgt_vocab_size, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
        Helper: Construct a model from hyperparameters.
    :param src_vocab_size:
    :param tgt_vocab_size:
    :param N:
    :param d_model: word_emb_size, all the middle dimensions
    :param d_ff: dim in the middle of feed forward layer
    :param h: number of heads
    :param dropout:
    :return:
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    pos_op = AddPositionalEncoding(d_model, dropout)
    model = Transformer(
        StackedEncoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        StackedDecoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(WeightedEmbeddings(d_model, src_vocab_size), c(pos_op)),
        nn.Sequential(WeightedEmbeddings(d_model, tgt_vocab_size), c(pos_op)),
        OutputLayer(d_model, tgt_vocab_size))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        # inputs of model are attributes of Batch object
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1  # 都从1开始
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(DataUtil.subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


if __name__ == "__main__":
    # Train the simple copy task.
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_input_embed_op[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    mode = "train"
    model_path = "model.p"
    if mode == "train":
        for epoch in range(10):
            model.train()
            run_epoch(data_gen(V, 30, 20), model,
                      SimpleLossCompute(model.output_layer, criterion, model_opt))
            model.eval()
            print(run_epoch(data_gen(V, 30, 5), model,
                            SimpleLossCompute(model.output_layer, criterion, None)))
        pickle.dump(model, open(model_path, "wb"))

    elif mode == "eval":
        # model.eval()
        if not os.path.isfile(model_path):
            print("Should train before eval!")
        else:
            model = pickle.load(open(model_path, "rb"))
            src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
            src_mask = Variable(torch.ones(1, 1, 10))
            print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

    else:
        pass
