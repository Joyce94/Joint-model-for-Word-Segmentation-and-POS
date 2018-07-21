import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import numpy
torch.manual_seed(123)
random.seed(123)

class Encode(nn.Module):
    def __init__(self, config, params):
        super(Encode, self).__init__()
        self.word_num = params.word_num
        self.pos_num = params.pos_num
        self.char_num = params.char_num
        self.bichar_num = params.bichar_num
        self.extchar_num = params.extchar_num
        self.extbichar_num = params.extbichar_num
        self.char_type_num = params.char_type_num
        self.wordlen_num = params.wordlen_num

        # self.id2word = params.word_alphabet.id2word
        # self.word2id = params.word_alphabet.word2id
        self.wordPadID = params.wordPadID
        self.charPadID = params.charPadID
        self.bicharPadID = params.bicharPadID
        self.charTypePadID = params.charTypePadID
        self.wordlenPadID = params.wordlenPadID

        self.use_cuda = params.use_cuda
        self.feature_count = config.shrink_feature_thresholds

        self.word_dims = config.word_dims
        self.char_dims = config.char_dims
        self.bichar_dims = config.bichar_dims
        self.char_type_dims = config.char_type_dims
        self.wordlen_dims = config.wordlen_dims

        self.lstm_hiddens = config.lstm_hiddens
        self.linear_hiddens = config.linear_hiddens

        self.dropout_emb = nn.Dropout(p=config.dropout_emb)
        self.dropout_lstm = nn.Dropout(p=config.dropout_lstm)

        self.lstm_layers = config.lstm_layers
        self.batch_size = config.train_batch_size

        self.char_emb = nn.Embedding(self.char_num, self.char_dims)
        self.char_emb.weight.requires_grad = True

        self.extchar_emb = nn.Embedding(self.extchar_num, self.char_dims)
        self.extchar_emb.weight.requires_grad = False

        self.bichar_emb = nn.Embedding(self.bichar_num, self.bichar_dims)
        self.bichar_emb.weight.requires_grad = True

        self.extbichar_emb = nn.Embedding(self.extbichar_num, self.bichar_dims)
        self.extbichar_emb.weight.requires_grad = False

        self.char_type_emb = nn.Embedding(self.char_type_num, self.char_type_dims)
        self.char_type_emb.weight.requires_grad = True

        self.wordlen_emb = nn.Embedding(self.wordlen_num, self.wordlen_dims)
        self.wordlen_emb.weight.requires_grad = True

        # static
        if params.pretrain_char_embedding is not None:
            pretrain_weight = torch.FloatTensor(params.pretrain_char_embedding)
            self.extchar_emb.weight.data.copy_(pretrain_weight)
        if params.pretrain_bichar_embedding is not None:
            pretrain_weight = torch.FloatTensor(params.pretrain_bichar_embedding)
            self.extbichar_emb.weight.data.copy_(pretrain_weight)

        # finetune
        # for idx in range(self.char_dims):
        #     self.char_emb.weight.data[self.charPadID][idx] = 0
        # # another method
        # # char_pad = torch.zeros(self.char_dims)
        # # self.char_emb.weight.data[self.charPadID] = char_pad
        # for idx in range(self.bichar_dims):
        #     self.bichar_emb.weight.data[self.bicharPadID][idx] = 0
        # for idx in range(self.char_type_dims):
        #     self.char_type_emb.weight.data[self.charTypePadID][idx] = 0
        # # for idx in range(self.wordlen_dims):
        # #     self.wordlen_emb.weight.data[self.wordlenPadID][idx] = 0

        self.dropout_emb = nn.Dropout(config.dropout_emb)
        self.input_dims = self.char_dims*2+self.bichar_dims*2+self.char_type_dims

        self.fc = nn.Linear(self.input_dims, self.linear_hiddens)
        nn.init.xavier_uniform(self.fc.weight)
        self.fc.bias.data.uniform_(-numpy.sqrt(6 / (self.linear_hiddens + 1)),
                                           numpy.sqrt(6 / (self.linear_hiddens + 1)))

        # self.lstm_left = nn.LSTM(self.linear_hiddens, self.lstm_hiddens, dropout=config.dropout_lstm)
        # self.lstm_right = nn.LSTM(self.linear_hiddens, self.lstm_hiddens, dropout=config.dropout_lstm)

        self.lstm_left = nn.LSTMCell(input_size=self.linear_hiddens, hidden_size=self.lstm_hiddens, bias=True)

        self.lstm_right = nn.LSTMCell(input_size=self.linear_hiddens, hidden_size=self.lstm_hiddens, bias=True)

        # self.init_lstm(self.lstm_left)
        # self.init_lstm(self.lstm_right)

        nn.init.xavier_uniform(self.lstm_left.weight_ih)
        nn.init.xavier_uniform(self.lstm_left.weight_hh)
        value = np.sqrt(6 / (self.lstm_hiddens + 1))
        self.lstm_left.bias_ih.data.uniform_(-value, value)
        self.lstm_left.bias_hh.data.uniform_(-value, value)

        nn.init.xavier_uniform(self.lstm_right.weight_ih)
        nn.init.xavier_uniform(self.lstm_right.weight_hh)
        self.lstm_right.bias_ih.data.uniform_(-value, value)
        self.lstm_right.bias_hh.data.uniform_(-value, value)
        # init lstm weight and bias
        # nn.init.xavier_uniform(self.lstm_left.weight_ih_l0)
        # nn.init.xavier_uniform(self.lstm_left.weight_hh_l0)
        # nn.init.xavier_uniform(self.lstm_right.weight_ih_l0)
        # nn.init.xavier_uniform(self.lstm_right.weight_hh_l0)
        # value = np.sqrt(6 / (self.lstm_hiddens + 1))
        # self.lstm_left.bias_ih_l0.data.uniform_(-value, value)
        # self.lstm_left.bias_hh_l0.data.uniform_(-value, value)
        # self.lstm_right.bias_ih_l0.data.uniform_(-value, value)
        # self.lstm_right.bias_hh_l0.data.uniform_(-value, value)


    def init_cell_hidden(self, batch_size):
        if self.use_cuda:
            return (Variable(torch.zeros(batch_size, self.lstm_hiddens)).cuda(),
                    Variable(torch.zeros(batch_size, self.lstm_hiddens)).cuda())
        else:
            return (Variable(torch.zeros(batch_size, self.lstm_hiddens)),
                    Variable(torch.zeros(batch_size, self.lstm_hiddens)))

    def init_hidden(self, batch_size):
        if self.use_cuda:
            return (Variable(torch.zeros(batch_size, self.lstm_hiddens)).cuda(),
                     Variable(torch.zeros(batch_size, self.lstm_hiddens)).cuda())
        else:
            return (Variable(torch.zeros(batch_size, self.lstm_hiddens)),
                     Variable(torch.zeros(batch_size, self.lstm_hiddens)))

    def init_lstm(self, input_lstm):
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l'+str(ind))
            bias = np.sqrt(6.0/(weight.size(0)+weight.size(1)))
            nn.init.uniform(weight, -bias, bias)
        if input_lstm.bias:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.bias_ih_l'+str(ind))
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2*input_lstm.hidden_size]=1

    def forward(self, var_b, list_b, mask_v, length):
        chars_var = var_b[0]
        left_bichar_var = var_b[1]
        right_bichar_var = var_b[2]
        extchars_var = var_b[3]
        extleft_bichar_var = var_b[4]
        extright_bichar_var = var_b[5]
        char_type_var = var_b[6]

        char_emb = self.dropout_emb(self.char_emb(chars_var))
        ##### char_emb: variable(batch_size, max_length, char_dims)
        extchar_emb = self.dropout_emb(self.extchar_emb(extchars_var))
        left_bichar_emb = self.dropout_emb(self.bichar_emb(left_bichar_var))
        extleft_bichar_emb = self.dropout_emb(self.extbichar_emb(extleft_bichar_var))
        right_bichar_emb = self.dropout_emb(self.bichar_emb(right_bichar_var))
        extright_bichar_emb = self.dropout_emb(self.extbichar_emb(extright_bichar_var))
        char_type_emb = self.dropout_emb(self.char_type_emb(char_type_var))

        batch_size = char_emb.size(0)
        max_length = char_emb.size(1)

        left_total = torch.cat([char_emb, extchar_emb, left_bichar_emb, extleft_bichar_emb, char_type_emb], 2)
        right_total = torch.cat([char_emb, extchar_emb, right_bichar_emb, extright_bichar_emb, char_type_emb], 2)

        left_non_linear = torch.transpose(self.dropout_lstm(F.tanh(self.fc(left_total))), 0, 1)
        right_non_linear = torch.transpose(self.dropout_lstm(F.tanh(self.fc(right_total))), 0, 1)
        ##### left_non_linear: variable(max_length, batch_size, linear_hiddens)
        ##### right_non_linear: variable(max_length, batch_size, linear_hiddens)

        # left_lstm_out, _ = self.lstm_left(left_non_linear)
        # right_lstm_out, _ = self.lstm_right(right_non_linear)
        # ##### left_lstm_out: variable(max_length, barch_size, lstm_hiddens)
        # ##### right_lstm_out: variable(max_length, barch_size, lstm_hiddens)
        #
        # left_lstm_out = torch.transpose(left_lstm_out, 0, 1)
        # right_lstm_out = torch.transpose(right_lstm_out, 0, 1)
        # for id in range(batch_size):
        #     idx = [i for i in range(length[id]-1, -1, -1)]
        #     if len(idx) != max_length:
        #         idx = idx + list(range(length[id], max_length))
        #     idx = torch.LongTensor(idx)
        #     if self.use_cuda: idx = idx.cuda()
        #     right_lstm_out[id].data = right_lstm_out[id].data.index_select(0, idx)


        left_h, left_c = self.init_cell_hidden(batch_size)
        left_lstm_out = []
        for idx in range(max_length):
            left_h, left_c = self.lstm_left(left_non_linear[idx], (left_h, left_c))
            left_h = self.dropout_lstm(left_h)
            left_lstm_out.append(left_h.view(batch_size, 1, self.lstm_hiddens))
        left_lstm_out = torch.cat(left_lstm_out, 1)

        right_h , right_c = self.init_cell_hidden(batch_size)
        right_lstm_out = []
        for idx in reversed(range(max_length)):
            right_h, right_c = self.lstm_right(right_non_linear[idx], (right_h, right_c))
            right_h = self.dropout_lstm(right_h)
            right_lstm_out.insert(0, right_h.view(batch_size, 1, self.lstm_hiddens))
        right_lstm_out = torch.cat(right_lstm_out, 1)


        lstm_out = torch.cat([left_lstm_out, right_lstm_out], 2)
        # lstm_out: variable(batch_size, max_length, 2*lstm_hiddens)
        return lstm_out















