import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F

import args
import models.Attention


def translate(result, dict):
    print(' '.join([dict.idx2word[i] for i in result[1:-1]]))


class RNNEncoder(nn.Module):

    def __init__(self, vocab_size, vocab, dropout=0.3):
        super(RNNEncoder, self).__init__()

        self.no_pack_padded_seq = False
        self.hidden_size = args.HIDDEN_SIZE

        self.rnn = nn.LSTM(input_size=args.WORD_VEC_SIZE,
                           hidden_size=args.HIDDEN_SIZE,
                           dropout=dropout,
                           batch_first=True,
                           bidirectional=True)
        self.embeddings = nn.Embedding(vocab_size, args.WORD_VEC_SIZE, padding_idx=args.PAD_TOKEN)
        self.embeddings.weight.data.normal_(mean=0, std=0.1)
        self.vocab = vocab

    def forward(self, input, lengths=None, hidden=None):
        emb = self.embeddings(input)
        packed_emb = emb

        if lengths is not None and not self.no_pack_padded_seq:
            lengths, index = lengths.sort(0, descending=True)

            if args.USE_CUDA:
                index = index.cuda()

            emb = emb[index]
            packed_emb = pack(emb, lengths.numpy(), batch_first=True)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs, batch_first=True)[0]
            _, index = index.sort(0)
            outputs = outputs[index]

        return hidden_t, outputs


class RNNDecoder(nn.Module):

    def __init__(self, weight, vocab_size, vocab, dropout=0.3):
        super(RNNDecoder, self).__init__()

        self.hidden_size = args.HIDDEN_SIZE

        self.rnn = nn.LSTM(input_size=args.WORD_VEC_SIZE,
                           hidden_size=args.HIDDEN_SIZE * 2,
                           dropout=dropout,
                           batch_first=True)
        self.attn = models.Attention.Attention(args.HIDDEN_SIZE * 2)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.hidden_size * 2, vocab_size, bias=True)
        self.embeddings = nn.Embedding(vocab_size, args.WORD_VEC_SIZE, padding_idx=args.PAD_TOKEN)
        self.embeddings.weight = weight

        torch.nn.init.xavier_uniform(self.out.weight)

        self.vocab = vocab

    def forward(self, input, context, hidden, test=False, context_lengths=None):
        if not test:
            emb = self.embeddings(input[:, :-1])
        else:
            emb = self.embeddings(input)

        rnn_output, hidden = self.rnn(emb, hidden)

        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(rnn_output.contiguous(), context, context_lengths=context_lengths)

        outputs = self.dropout(attn_outputs)

        return F.log_softmax(self.out(outputs), dim=2), hidden, rnn_output, emb

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, enc_hidden):
        return RNNDecoderState(tuple([self._fix_enc_hidden(enc_hidden[i])
                                      for i in range(len(enc_hidden))]))


class DocSumModel(nn.Module):

    def __init__(self, encoder, decoder, vocab):
        super(DocSumModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab

    def forward(self, src, tgt, lengths, test=False):
        enc_hidden, context = self.encoder(src, lengths)

        if not test:
            enc_state = self.decoder.init_decoder_state(enc_hidden)
            out, dec_state, rnn_outputs, src_emb = self.decoder(
                tgt, context, enc_state.hidden)
        else:
            dec_state = self.decoder.init_decoder_state(enc_hidden).hidden
            decoder_input = Variable(torch.LongTensor([[args.SOS_TOKEN]]))
            result = [args.SOS_TOKEN]
            i = 0

            while True:

                if i == args.ABSTRACT_LENGTH_THRESHOLD:
                    result.append(args.EOS_TOKEN)
                    break

                out, dec_state, rnn_outputs, src_emb = self.decoder(
                    decoder_input, context, dec_state, test=True)

                topv, topi = out.data.topk(1)
                ni = topi[0][0]
                ni = np.asscalar(ni.numpy()[0])
                result.append(ni)

                if ni == args.EOS_TOKEN:
                    break

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if args.USE_CUDA else decoder_input

                i += 1

            translate(result, self.vocab)

            translate(tgt.numpy()[0], self.vocab)
            print()

        return out, dec_state


class RNNDecoderState(object):
    def __init__(self, rnnstate):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate

    @property
    def _all(self):
        return self.hidden

    def detach(self):
        """
        Detaches all Variables from the graph
        that created it, making it a leaf.
        """
        for h in self._all:
            if h is not None:
                h.detach()


