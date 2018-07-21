import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from model.state import state_nowordlstm
from model.state import state_batch
from model.state import state
import data.utils as utils
import numpy
import re

torch.manual_seed(123)
random.seed(123)


class Decode(nn.Module):
    def __init__(self, config, params):
        super(Decode, self).__init__()
        self.word_num = params.word_num
        self.pos_num = params.pos_num
        self.segpos_num = params.segpos_num
        self.char_num = params.char_num
        self.bichar_num = params.bichar_num
        self.extchar_num = params.extchar_num
        self.extbichar_num = params.extbichar_num
        self.char_type_num = params.char_type_num
        self.wordlen_num = params.wordlen_num
        self.wordlen_max = params.wordlen_max

        self.id2pos = params.pos_alphabet.id2word
        self.pos2id = params.pos_alphabet.word2id
        self.id2wordlen = params.wordlen_alphabet.id2word
        self.wordlen2id = params.wordlen_alphabet.word2id
        self.id2gold = params.segpos_alphabet.id2word
        self.gold2id = params.segpos_alphabet.word2id

        self.wordPadID = params.wordPadID
        self.charPadID = params.charPadID
        self.bicharPadID = params.bicharPadID
        self.charTypePadID = params.charTypePadID
        self.wordlenPadID = params.wordlenPadID
        self.posPadID = params.posPadID
        self.appID = params.appID
        self.actionPadID = params.actionPadID

        self.use_cuda = params.use_cuda
        self.feature_count = config.shrink_feature_thresholds

        self.word_dims = config.word_dims
        self.char_dims = config.char_dims
        self.bichar_dims = config.bichar_dims
        self.char_type_dims = config.char_type_dims
        self.wordlen_dims = config.wordlen_dims
        self.pos_dims = config.pos_dims

        self.lstm_hiddens = config.lstm_hiddens
        self.linear_hiddens = config.linear_hiddens

        self.dropout_emb = nn.Dropout(p=config.dropout_emb)
        self.dropout_lstm = nn.Dropout(p=config.dropout_lstm)

        self.lstm_layers = config.lstm_layers
        self.batch_size = config.train_batch_size

        self.wordlen_emb = nn.Embedding(self.wordlen_num, self.wordlen_dims)
        self.wordlen_emb.weight.requires_grad = True
        self.pos_emb = nn.Embedding(self.pos_num, self.pos_dims)
        self.pos_emb.weight.requires_grad = True

        # finetune
        for idx in range(self.pos_dims):
            self.pos_emb.weight.data[self.posPadID][idx] = 0
        for idx in range(self.wordlen_dims):
            self.wordlen_emb.weight.data[self.wordlenPadID][idx] = 0

        nn.init.uniform(self.wordlen_emb.weight, a=-numpy.sqrt(3 / self.wordlen_dims), b=numpy.sqrt(3 / self.wordlen_dims))
        nn.init.uniform(self.pos_emb.weight, a=-numpy.sqrt(3 / self.pos_dims), b=numpy.sqrt(3 / self.pos_dims))

        self.dropout_emb = nn.Dropout(config.dropout_emb)

        self.lstmcell = nn.LSTMCell(self.linear_hiddens, self.lstm_hiddens, bias=True)
        nn.init.xavier_uniform(self.lstmcell.weight_ih)
        nn.init.xavier_uniform(self.lstmcell.weight_hh)
        self.lstmcell.bias_hh.data.uniform_(-np.sqrt(6 / (self.lstm_hiddens + 1)),
                                            np.sqrt(6 / (self.lstm_hiddens + 1)))
        self.lstmcell.bias_ih.data.uniform_(-np.sqrt(6 / (self.lstm_hiddens + 1)),
                                            np.sqrt(6 / (self.lstm_hiddens + 1)))

        self.input_dims = self.pos_dims + self.wordlen_dims + 2*self.lstm_hiddens
        self.fc = nn.Linear(self.input_dims, self.linear_hiddens, bias=True)
        self.combine = nn.Linear(self.lstm_hiddens*3, self.segpos_num, bias=False)
        nn.init.xavier_uniform(self.fc.weight)
        nn.init.xavier_uniform(self.combine.weight)

        self.fc.bias.data.uniform_(-np.sqrt(6 / (self.linear_hiddens + 1)),
                                               np.sqrt(6 / (self.linear_hiddens + 1)))

        nn.init.xavier_uniform(self.fc.weight)
        self.fc.bias.data.uniform_(-numpy.sqrt(6 / (self.lstm_hiddens + 1)),
                                               numpy.sqrt(6 / (self.lstm_hiddens + 1)))

    def init_hidden_cell(self):
        if self.use_cuda:
            return (Variable(torch.zeros(1, self.lstm_hiddens)).cuda(),
                    Variable(torch.zeros(1, self.lstm_hiddens)).cuda())
        else:
            return (Variable(torch.zeros(1, self.lstm_hiddens)),
                    Variable(torch.zeros(1, self.lstm_hiddens)))


    def forward(self, encoder_output, var_b, list_b, mask_v, length, is_train=True):
        chars_var = var_b[0]
        batch_size = chars_var.size(0)
        max_length = chars_var.size(1)
        # print(mask_v)       # [torch.ByteTensor of size 16x137]

        # print('length:', length)        # length: [137, 100, 98, 60, 49]
        # wordlen = list_b[0]
        gold = list_b[1]
        # pos = list_b[2]
        chars = list_b[3]

        batch_output = []
        batch_state = []
        for idx in range(batch_size):
            s = state(gold[idx], chars[idx])
            sent_output = []
            length_cur = length[idx]
            for idy in range(max_length):
                if idy < length_cur:
                    if idy == 0: h, c = self.init_hidden_cell()
                    h, c = self.wordlstm_cell(s, h, c, idy, encoder_output[idx])
                    v = torch.cat((h, encoder_output[idx][idy].view(1, self.lstm_hiddens * 2)), 1)

                    output = self.combine(v)
                    # print(output)
                    if idy == 0:
                        output.data[0][self.appID] = -10e+99
                    self.my_action(s, idy, output, h, c, is_train)
                    sent_output.append(output)
                else:
                    assist_add = Variable(torch.zeros(1, self.segpos_num)).type(torch.FloatTensor)
                    if self.use_cuda: assist_add = assist_add.cuda()
                    sent_output.append(assist_add)
            sent_output = torch.cat(sent_output, 0)
            batch_output.append(sent_output)
            batch_state.append(s)
        batch_output = torch.cat(batch_output, 0)
        # print(batch_output)
        return batch_output, batch_state


    def wordlstm_cell(self, state, h, c, index, encode_out):
        if index == 0:
            z = Variable(torch.zeros(1, self.lstm_hiddens))
            if self.use_cuda: z = z.cuda()
        else:
            last_pos = Variable(torch.zeros(1)).type(torch.LongTensor)
            if self.use_cuda: last_pos = last_pos.cuda()
            last_pos.data[0] = state.pos_id[-1]
            # print(last_pos)
            last_pos_emb = self.dropout_emb(self.pos_emb(last_pos))

            last_word_len = len(state.words[-1])
            start = index - last_word_len
            end = index
            chars_emb = []

            for idx in range(start, end):
                chars_emb.append(encode_out[idx].view(1, 1, 2 * self.lstm_hiddens))
            chars_emb = torch.cat(chars_emb, 1)
            last_word_emb = F.avg_pool1d(chars_emb.permute(0, 2, 1), last_word_len).view(1, self.lstm_hiddens * 2)

            if last_word_len > 6: last_word_len = 7

            word_len_id = self.wordlen2id[last_word_len]
            word_len = Variable(torch.zeros(1)).type(torch.LongTensor)
            word_len.data[0] = word_len_id
            if self.use_cuda: word_len = word_len.cuda()
            word_len_emb = self.wordlen_emb(word_len)
            emb_total = torch.cat((last_pos_emb, last_word_emb, word_len_emb), 1)
            z = self.dropout_lstm(F.tanh(self.fc(emb_total)))

        h, c = self.lstmcell(z, (h, c))
        return h, c


    def my_action(self, state, index, output, h_now, c_now, is_train):
        if is_train:
            action = state.m_gold[index]
        else:
            _, action_id = torch.max(output, dim=1)
            action_id = action_id.data.tolist()[0]
            action = self.id2gold[action_id]
        state.actions.append(action)

        pos = action.find('#')
        if pos == -1:
            ###app
            # print(state.words[-1])
            state.words[-1] += state.m_chars[index]
            # print(state.words[-1])
        else:
            ###sep
            tmp_word = state.m_chars[index]
            state.words.append(tmp_word)
            posLabel = action[pos + 1:]
            state.pos_labels.append(posLabel)
            posID = self.pos2id[posLabel]
            state.pos_id.append(posID)
            state.word_cells.append(c_now)
            state.word_hiddens.append(h_now)






