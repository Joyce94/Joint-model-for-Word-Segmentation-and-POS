import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from model.state import state
import data.utils as utils

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

        self.dropout_emb = nn.Dropout(config.dropout_emb)
        self.input_dims = self.pos_dims + self.wordlen_dims + 2*self.lstm_hiddens
        self.fc = nn.Linear(self.input_dims, self.linear_hiddens)
        self.lstmcell = nn.LSTMCell(self.linear_hiddens, self.lstm_hiddens)

        self.combine = nn.Linear(self.lstm_hiddens*3, self.segpos_num)



    def init_hidden_cell(self):
        if self.use_cuda:
            return (Variable(torch.zeros(1, self.lstm_hiddens)).cuda(),
                    Variable(torch.zeros(1, self.lstm_hiddens)).cuda())
        else:
            return (Variable(torch.zeros(1, self.lstm_hiddens)),
                    Variable(torch.zeros(1, self.lstm_hiddens)))

    # def init_cell_hidden(self, batch_size):
    #     if self.use_cuda:
    #         return (Variable(torch.zeros(batch_size, self.lstm_hiddens)).cuda(),
    #                 Variable(torch.zeros(batch_size, self.lstm_hiddens)).cuda())
    #     else:
    #         return (Variable(torch.zeros(batch_size, self.lstm_hiddens)),
    #                 Variable(torch.zeros(batch_size, self.lstm_hiddens)))

    def forward(self, encode_out, var_b, list_b, mask_v, length):
        chars_var = var_b[0]
        batch_size = chars_var.size(0)
        max_length = chars_var.size(1)

        wordlen = list_b[0]
        gold = list_b[1]
        pos = list_b[2]
        chars = list_b[3]

        output_total = []
        for idx in range(batch_size):
            wordlen_sen = wordlen[idx]
            gold_sen = gold[idx]
            pos_sen = pos[idx]
            chars_sen = chars[idx]
            ##### len(wordlen_sen) == len(pos_sen)
            ##### len(gold_sen) == len(chars_sen)
            sent_output = []

            sent = state(gold_sen, chars_sen)
            h, c = self.init_hidden_cell()
            z = Variable(torch.zeros(1, self.lstm_hiddens))
            if self.use_cuda: z = z.cuda()
            for idy in range(max_length):
                if idy < len(sent.chars):
                    h, c, z = self.wordlstm_cell(sent, encode_out[idx], idy, h, c, z)
                    ##### h: variable (1, lstm_hiddens)
                    ##### encode_out[idx][idy]: variable (2*lstm_hiddens)
                    v = torch.cat([h, encode_out[idx][idy].unsqueeze(0)], 1)
                    output = self.combine(v)
                    ##### output: variable (1, segpos_num)

                    ##### remove the probability of predicting 'app' in the beginning
                    if idy == 0: output.data[0][self.appID] = -10e+99
                    self.action(sent, idy, output, train=True)
                    sent_output.append(output)
                else:
                    add_pad = Variable(torch.zeros(1, self.segpos_num))
                    if self.use_cuda: add_pad = add_pad.cuda()
                    sent_output.append(add_pad)
            sent_output = torch.cat(sent_output, 0)
            ##### sent_output: variable (max_length, segpos_num)
            output_total.append(sent_output.unsqueeze(0))
        output_total = torch.cat(output_total, 0)
        ##### output_total: variable (batch_size, max_length, segpos_num)
        return output_total


    def wordlstm_cell(self, sent, encode_sent, index, h, c, z):
        if index == 0:
            sent.words_record.append(sent.chars[index])
            sent.pos_record.append(sent.gold[index].split('#')[1])
            sent.pos_index.append(self.pos2id[sent.gold[index].split('#')[1]])
            # print(sent.gold[index].split('#')[1])
            # print(self.pos2id[sent.gold[index].split('#')[1]])
        else:
            last_pos = Variable(torch.LongTensor([sent.pos_index[-1]]))
            if self.use_cuda: last_pos = last_pos.cuda()
            last_pos_emb = self.pos_emb(last_pos)
            ##### last_pos_emb: variable (1, pos_dims)

            last_wordlen = len(sent.words_record[-1])
            last_wordlen_id = self.wordlen2id[last_wordlen]
            last_wordlen_var = Variable(torch.LongTensor([last_wordlen_id]))
            if self.use_cuda: last_wordlen_var = last_wordlen_var.cuda()
            last_wordlen_emb = self.wordlen_emb(last_wordlen_var)
            ##### last_wordlen_emb: variable (1, wordlen_dims)

            start = index-last_wordlen
            last_word_emb = torch.mean(encode_sent[start:index], 0).unsqueeze(0)
            ##### last_word_emb: variable (1, 2*lstm_hiddens)
            z_in = torch.cat([last_word_emb, last_wordlen_emb, last_pos_emb], 1)
            ##### z: variable (1, pos_dims+wordlen_dims+2*lstm_hiddens)

            z = F.tanh(self.fc(z_in))
        h, c = self.lstmcell(z, (h, c))

        return h, c, z

    def action(self, sent, index, output, train=True):
        if train:
            action = sent.gold[index]
        else:
            max_score, max_index = torch.max(output, dim=1)
            # print(max_index)
            # print(max_score)
            action = self.id2gold[utils.to_scalar(max_index)]

        sent.action.append(action)
        pos_id = action.find('#')
        if pos_id == -1:
            last_word_record = sent.words_record[-1] + sent.chars[index]
            sent.words_record[-1] = last_word_record
        else:
            sent.words_record.append(sent.chars[index])
            pos_record = action[(pos_id+1):]
            sent.pos_record.append(pos_record)
            sent.pos_index.append(self.pos2id[pos_record])


