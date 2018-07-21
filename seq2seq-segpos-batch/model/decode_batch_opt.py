import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from model.state import state
from model.state import state_batch
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
        # self.batch_size = config.train_batch_size

        self.wordlen_emb = nn.Embedding(self.wordlen_num, self.wordlen_dims)
        self.wordlen_emb.weight.requires_grad = True
        self.pos_emb = nn.Embedding(self.pos_num, self.pos_dims)
        self.pos_emb.weight.requires_grad = True

        # finetune
        # for idx in range(self.pos_dims):
        #     self.pos_emb.weight.data[self.posPadID][idx] = 0
        # for idx in range(self.wordlen_dims):
        #     self.wordlen_emb.weight.data[self.wordlenPadID][idx] = 0

        nn.init.uniform(self.wordlen_emb.weight, a=-numpy.sqrt(3 / self.wordlen_dims), b=numpy.sqrt(3 / self.wordlen_dims))
        nn.init.uniform(self.pos_emb.weight, a=-numpy.sqrt(3 / self.pos_dims), b=numpy.sqrt(3 / self.pos_dims))

        self.dropout_emb = nn.Dropout(config.dropout_emb)
        self.input_dims = self.pos_dims + self.wordlen_dims + 2*self.lstm_hiddens

        self.lstmcell = nn.LSTMCell(self.linear_hiddens, self.lstm_hiddens)
        nn.init.xavier_uniform(self.lstmcell.weight_ih)
        nn.init.xavier_uniform(self.lstmcell.weight_hh)
        self.lstmcell.bias_hh.data.uniform_(-np.sqrt(6 / (self.lstm_hiddens + 1)),
                                            np.sqrt(6 / (self.lstm_hiddens + 1)))
        self.lstmcell.bias_ih.data.uniform_(-np.sqrt(6 / (self.lstm_hiddens + 1)),
                                            np.sqrt(6 / (self.lstm_hiddens + 1)))

        self.fc = nn.Linear(self.input_dims, self.linear_hiddens)
        self.combine = nn.Linear(self.lstm_hiddens*3, self.segpos_num, bias=False)
        nn.init.xavier_uniform(self.fc.weight)
        nn.init.xavier_uniform(self.combine.weight)

        self.fc.bias.data.uniform_(-np.sqrt(6 / (self.linear_hiddens + 1)), np.sqrt(6 / (self.linear_hiddens + 1)))
        self.softmax = nn.LogSoftmax()

    def init_hidden_cell_batch (self, batch_size):
        if self.use_cuda:
            return (Variable(torch.zeros(batch_size, self.lstm_hiddens)).cuda(),
                    Variable(torch.zeros(batch_size, self.lstm_hiddens)).cuda())
        else:
            return (Variable(torch.zeros(batch_size, self.lstm_hiddens)),
                    Variable(torch.zeros(batch_size, self.lstm_hiddens)))

    def forward(self, encode_out, var_b, list_b, mask_v, length):
        chars_var = var_b[0]
        batch_size = chars_var.size(0)
        max_length = chars_var.size(1)
        # print(mask_v)       # [torch.ByteTensor of size 16x137]
        # print('length:', length)        # length: [137, 100, 98, 60, 49]
        # wordlen = list_b[0]
        gold = list_b[1]
        # pos = list_b[2]
        chars = list_b[3]
        output_total = []
        encode_out = encode_out.transpose(0, 1)
        ##### encode_out: variable(max_len, batch_size, 2*lstm_hiddens)
        masks = mask_v.transpose(0, 1)

        states = state_batch(gold, chars, batch_size, max_length)
        for idx in range(max_length):
            if idx == 0: h, c = self.init_hidden_cell_batch(batch_size)
            h, c = self.wordlstm_cell_batch(idx, states, encode_out, mask_v, h, c, batch_size)
            ##### h: variable (batch_size, lstm_hiddens)
            ##### encode_out[idx]: variable (batch_size, 2*lstm_hiddens)
            v = torch.cat([h, encode_out[idx]], 1)
            output = self.combine(v)
            ##### output: variable (batch_size, segpos_num)
            ##### remove the probability of predicting 'app' in the beginning

            ##### 不知道为啥这样才是对的，保险起见换成速度慢一点但是绝对是正确的
            if idx == 0:
                output = output.transpose(0, 1)             ###### 好像只能这样并行才是对的
                output.data[:][self.appID] = -10e+99
                output = output.transpose(0, 1)
            output = output.transpose(0, 1)
            output.data[:][self.actionPadID] = -10e+99
            output = output.transpose(0, 1)
            # if idx == 0:
            #     for idy in range(batch_size):
            #         output.data[idy][self.appID] = -10e+99
            #         output.data[idy][self.actionPadID] = -10e+99
            # else:
            #     for idy in range(batch_size):
            #         output.data[idy][self.actionPadID] = -10e+99

            self.action_batch(idx, states, output, length)

            # correct output
            mask = masks[idx]       # [torch.ByteTensor of size 16]
            mask_broadcast = mask.unsqueeze(1).expand(batch_size, self.segpos_num)
            # assist_var = Variable(torch.zeros(batch_size, self.segpos_num)).type(torch.FloatTensor)
            assist_var = Variable(torch.FloatTensor([[-10e+99]*self.segpos_num]*batch_size))
            if self.use_cuda: assist_var = assist_var.cuda()
            output_c = assist_var.masked_scatter_(mask_broadcast, output)
            output_total.append(output_c.unsqueeze(0))
        output_total = torch.cat(output_total, 0)
        # output_total = output_total.view(output_total.size(0)*output_total.size(1), output_total.size(2))
        # output_total = self.softmax(output_total)
        # print(output_total)
        ##### output_total: variable (batch_size, max_length, segpos_num)
        # print(states.action)
        return output_total, states


    def wordlstm_cell_batch(self, index, states, encode_out, masks, h, c, batch_size):
        if index == 0:
            # for id, ele in enumerate(states.chars[index]):
            #     states.words_record[id].append(ele)
            # for id, ele in enumerate(states.golds[index]):
            #     states.pos_record[id].append(ele.split('#')[1])
            # for id, ele in enumerate(states.golds[index]):
            #     states.pos_index[id].append(self.pos2id[ele.split('#')[1]])
            z = Variable(torch.zeros(batch_size, self.linear_hiddens))
            if self.use_cuda: z = z.cuda()
        else:
            last_pos_cur = [states.pos_index[id][-1] for id in range(batch_size)]
            # print(last_pos_cur)
            last_pos = Variable(torch.LongTensor(last_pos_cur))
            if self.use_cuda: last_pos = last_pos.cuda()
            # print(last_pos)
            # last_pos = pos_v[index-1]
            # print(last_pos)
            last_pos_emb = self.dropout_emb(self.pos_emb(last_pos))
            # print(last_pos_emb)
            ##### last_pos_emb: variable (batch_size, pos_dims)

            last_wordlen_id = []
            last_wordlen = []
            # for ele in states.words_record[-1]:
            for id in range(batch_size):
                # print(states.words_record[id][-1])
                word_record = states.words_record[id][-1]
                if '<pad>' in word_record:
                    word_record = re.sub('<pad>', '$', word_record)
                # if '<unk>' in word_record:
                #     print(word_record)
                ele_len = len(word_record)
                if ele_len > self.wordlen_max:
                    last_wordlen_id.append(self.wordlen2id[self.wordlen_max])
                    last_wordlen.append(self.wordlen_max)
                else:
                    # print(ele_len)
                    # print(self.wordlen2id)
                    last_wordlen_id.append(self.wordlen2id[ele_len])
                    last_wordlen.append(ele_len)

            last_wordlen_var = Variable(torch.LongTensor(last_wordlen_id))
            if self.use_cuda: last_wordlen_var = last_wordlen_var.cuda()
            last_wordlen_emb = self.wordlen_emb(last_wordlen_var)
            ##### last_wordlen_emb: variable (batch_size, wordlen_dims)

            ##### encode_out: variable(max_len, batch_size, 2*lstm_hiddens)
            # last_word_emb_sum = torch.sum(encode_out[:index], 0)
            encode_out = encode_out.transpose(0, 1)
            last_word_emb = []
            for idy in range(batch_size):
                start = index - last_wordlen[idy]
                # print(index)
                # print(last_wordlen)
                # print(last_wordlen[idy])
                last_word_emb.append(torch.mean(encode_out[idy][start:index], 0).unsqueeze(0))
            last_word_emb = torch.cat(last_word_emb, 0)
            ##### last_word_emb: variable (batch_size, 2*lstm_hiddens)
            # print(last_wordlen_emb)     # [torch.FloatTensor of size 10x20]
            # print(last_word_emb)        # [torch.FloatTensor of size 10x400]
            # print(last_pos_emb)         # [torch.FloatTensor of size 10x100]
            z_in = torch.cat([last_word_emb, last_wordlen_emb, last_pos_emb], 1)
            ##### z: variable (batch_size, pos_dims+wordlen_dims+2*lstm_hiddens)
            z = self.dropout_lstm(F.tanh(self.fc(z_in)))
            ##### z: variable (batch_size, linear_hiddens)
        h, c = self.lstmcell(z, (h, c))
        return h, c

    def action_batch(self, index, states, output, length):
        # if train:
        #     action = states.golds[index]        # ['SEP#P', 'SEP#CD', 'SEP#DT', 'SEP#NT', 'SEP#NN', 'SEP#P', 'SEP#NR', 'SEP#AD', 'SEP#NR', 'SEP#NN']
        #     if index == 0:
        #         for id, ele in enumerate(states.chars[index]):
        #             states.words_record[id].append(ele)
        #         # pos_record_cur = []
        #         # for ele in states.golds[index]:
        #         #     # states.pos_record[index].append(ele.split('#')[1])
        #         #     pos_record_cur.append(ele.split('#')[1])
        #         # states.pos_record.append(pos_record_cur)
        #         # # states.pos_record[index].append(states.gold[index].split('#')[1])
        #         for id, ele in enumerate(states.golds[index]):
        #             states.pos_record[id].append(ele.split('#')[1])
        #         # pos_index_cur = []
        #         # for ele in states.golds[index]:
        #         #     # states.pos_index[index].append(self.pos2id[ele.split('#')[1]])
        #         #     pos_index_cur.append(self.pos2id[ele.split('#')[1]])
        #         # states.pos_index.append(pos_index_cur)
        #         # # sent.pos_index.append(self.pos2id[sent.gold[index].split('#')[1]])
        #         for id, ele in enumerate(states.golds[index]):
        #             states.pos_index[id].append(self.pos2id[ele.split('#')[1]])
        # else:
        max_score, max_index = torch.max(output, dim=1)
        # print(max_index)
        # action = [self.id2gold[utils.to_scalar(ele)] for ele in max_index]
        action = []
        for id,ele in enumerate(max_index):
            action_cur = self.id2gold[utils.to_scalar(ele)]
            if index >= length[id]:
                action_cur = '<pad>'
            action.append(action_cur)
        # print(action)
        if index == 0:
            for id, ele in enumerate(action):
                pos_id = ele.find('#')
                if pos_id == -1:
                    print('action at the first index is error.')
                else:
                    states.words_record[id].append(states.chars[index][id])
                    pos_record = action[id][(pos_id+1):]
                    states.pos_record[id].append(pos_record)
                    pos_index = self.pos2id[pos_record]
                    states.pos_index[id].append(pos_index)
        # states.action.append(action)
        for id, ele in enumerate(action):
            states.action[id].append(ele)
        if index != 0:
            # last_words_record = states.words_record[index-1]
            # words_record_cur = []
            # pos_record_cur = []
            # pos_index_cur = []
            for id, ele in enumerate(action):
                if ele == '<pad>':
                    states.words_record[id].append('<pad>')
                    states.pos_record[id].append('<pad>')
                    states.pos_index[id].append(self.posPadID)
                else:
                    pos_id = ele.find('#')
                    if pos_id == -1:
                        last_cur = states.words_record[id][-1] + states.chars[index][id]    # chars和golds不一样
                        # words_record_cur.append(last_cur)
                        states.words_record[id][-1] = last_cur
                    else:
                        # words_record_cur.append(states.chars[index][id])
                        states.words_record[id].append(states.chars[index][id])
                        pos_record = action[id][(pos_id+1):]
                        # pos_record_cur.append(pos_record)
                        states.pos_record[id].append(pos_record)
                        pos_index = self.pos2id[pos_record]
                        # pos_index_cur.append(pos_index)
                        states.pos_index[id].append(pos_index)
            # states.words_record.append(words_record_cur)
            # states.pos_record.append(pos_record_cur)
            # states.pos_index.append(pos_index_cur)

    def forward_train(self, encode_out, var_b, list_b, mask_v, length):
        chars_var = var_b[0]
        batch_size = chars_var.size(0)
        max_length = chars_var.size(1)
        wordlen_var = var_b[-2]
        pos_var = var_b[-1]
        ##### wordlen_var, pos_var: variable(batch_size, max_length)

        # print(mask_v)       # [torch.ByteTensor of size 16x137]
        # print('length:', length)        # length: [137, 100, 98, 60, 49]
        # wordlen = list_b[0]
        gold = list_b[1]
        # pos = list_b[2]
        chars = list_b[3]
        output_total = []
        encode_out = encode_out.transpose(0, 1)
        ##### encode_out: variable(max_len, batch_size, 2*lstm_hiddens)
        masks = mask_v.transpose(0, 1)
        wordlen_var = wordlen_var.transpose(0, 1)
        pos_var = pos_var.transpose(0, 1)

        states = state_batch(gold, chars, batch_size, max_length)
        for idx in range(max_length):
            if idx == 0: h, c = self.init_hidden_cell_batch(batch_size)
            h, c = self.wordlstm_cell_batch_train(idx, states, encode_out, mask_v, h, c, batch_size, wordlen_var, pos_var)
            ##### h: variable (batch_size, lstm_hiddens)
            ##### encode_out[idx]: variable (batch_size, 2*lstm_hiddens)
            v = torch.cat([h, encode_out[idx]], 1)
            output = self.combine(v)
            ##### output: variable (batch_size, segpos_num)
            ##### remove the probability of predicting 'app' in the beginning
            # print(output)
            if idx == 0:
                output = output.transpose(0, 1)             ###### 好像只能这样并行才是对的
                output.data[:][self.appID] = -10e+99
                output = output.transpose(0, 1)
            output = output.transpose(0, 1)
            output.data[:][self.actionPadID] = -10e+99
            output = output.transpose(0, 1)
            # if idx == 0:
            #     for idy in range(batch_size):
            #         output.data[idy][self.appID] = -10e+99
            #         output.data[idy][self.actionPadID] = -10e+99
            # else:
            #     for idy in range(batch_size):
            #         output.data[idy][self.actionPadID] = -10e+99

            # self.action_batch_train(idx, states, output, length, train=True)

            # correct output
            mask = masks[idx]       # [torch.ByteTensor of size 16]
            mask_broadcast = mask.unsqueeze(1).expand(batch_size, self.segpos_num)
            # assist_var = Variable(torch.zeros(batch_size, self.segpos_num)).type(torch.FloatTensor)
            assist_var = Variable(torch.FloatTensor([[-10e+99] * self.segpos_num] * batch_size))
            if self.use_cuda: assist_var = assist_var.cuda()
            output_c = assist_var.masked_scatter_(mask_broadcast, output)
            output_total.append(output_c.unsqueeze(0))
        output_total = torch.cat(output_total, 0).transpose(0, 1)
        # print(output_total)
        # output_total = output_total.view(output_total.size(0)*output_total.size(1), output_total.size(2))
        # output_total = self.softmax(output_total)
        ##### output_total: variable (batch_size, max_length, segpos_num)
        # print(states.action)
        return output_total, states

    def wordlstm_cell_batch_train_check(self, index, states, encode_out, masks, h, c, batch_size, last_wordlen_v, last_pos_v):
        if index == 0:
            z = Variable(torch.zeros(batch_size, self.lstm_hiddens))
            if self.use_cuda: z = z.cuda()
        else:
            last_pos_cur = [states.pos_index[id][-1] for id in range(batch_size)]
            # print(last_pos_cur)
            last_pos = Variable(torch.LongTensor(last_pos_cur))
            if self.use_cuda: last_pos = last_pos.cuda()
            # print(last_pos)
            last_pos_c = last_pos_v[index]
            # print(last_pos_c)
            assert last_pos.data.tolist()[0] == last_pos_c.data.tolist()[0]
            last_pos_emb = self.pos_emb(last_pos)
            # print(last_pos_emb)
            ##### last_pos_emb: variable (batch_size, pos_dims)

            last_wordlen_id = []
            last_wordlen = []
            # for ele in states.words_record[-1]:
            for id in range(batch_size):
                # print(states.words_record[id][-1])
                word_record = states.words_record[id][-1]
                # print('391')
                # print(index)
                # print(id)
                # print(word_record)
                # if '<pad>' in word_record:
                #     word_record = re.sub('<pad>', '$', word_record)
                ele_len = len(word_record)
                # print(self.wordlen_max)
                if ele_len > self.wordlen_max:
                    last_wordlen_id.append(self.wordlen2id[self.wordlen_max+1])
                    last_wordlen.append(self.wordlen_max+1)
                else:
                    # print(ele_len)
                    # print(self.wordlen2id)
                    last_wordlen_id.append(self.wordlen2id[ele_len])
                    last_wordlen.append(ele_len)

            last_wordlen_var = Variable(torch.LongTensor(last_wordlen_id))
            if self.use_cuda: last_wordlen_var = last_wordlen_var.cuda()
            last_wordlen_c = last_wordlen_v[index]
            assert last_wordlen_c.data.tolist()[0] == last_wordlen_var.data.tolist()[0]
            # print('408')
            last_wordlen_emb = self.wordlen_emb(last_wordlen_var)
            ##### last_wordlen_emb: variable (batch_size, wordlen_dims)

            ##### encode_out: variable(max_len, batch_size, 2*lstm_hiddens)
            # last_word_emb_sum = torch.sum(encode_out[:index], 0)

            encode_out = encode_out.transpose(0, 1)
            last_word_emb = []
            for idy in range(batch_size):
                # print('420')
                # print(index)
                # print(idy)
                # print(states.words_record[id])
                last_wordlen_c_id = self.id2wordlen[last_wordlen_c[idy].data.tolist()[0]]
                if last_wordlen_c_id == '<pad>': last_wordlen_c_id = 1
                # print(last_wordlen_c_id)
                # print(last_wordlen[idy])
                # assert last_wordlen_c_id == last_wordlen[idy]

                start = index - last_wordlen[idy]
                # print(index)
                # print(last_wordlen)
                # print(last_wordlen[idy])
                last_word_emb.append(torch.mean(encode_out[idy][start:index], 0).unsqueeze(0))
            last_word_emb = torch.cat(last_word_emb, 0)
            ##### last_word_emb: variable (batch_size, 2*lstm_hiddens)
            # print(last_wordlen_emb)     # [torch.FloatTensor of size 10x20]
            # print(last_word_emb)        # [torch.FloatTensor of size 10x400]
            # print(last_pos_emb)         # [torch.FloatTensor of size 10x100]
            z_in = torch.cat([last_word_emb, last_wordlen_emb, last_pos_emb], 1)
            ##### z: variable (batch_size, pos_dims+wordlen_dims+2*lstm_hiddens)
            z = self.dropout_lstm(F.tanh(self.fc(z_in)))
            ##### z: variable (batch_size, linear_hiddens)
        h, c = self.lstmcell(z, (h, c))
        return h, c

    def wordlstm_cell_batch_train(self, index, states, encode_out, masks, h, c, batch_size, last_wordlen_v, last_pos_v):
        if index == 0:
            z = Variable(torch.zeros(batch_size, self.linear_hiddens))
            if self.use_cuda: z = z.cuda()
        else:
            last_pos_c = last_pos_v[index]
            last_pos_emb = self.dropout_emb(self.pos_emb(last_pos_c))
            last_wordlen_c = last_wordlen_v[index]
            last_wordlen_emb = self.wordlen_emb(last_wordlen_c)

            encode_out = encode_out.transpose(0, 1)
            last_word_emb = []
            for idy in range(batch_size):
                last_wordlen_c_len = self.id2wordlen[last_wordlen_c[idy].data.tolist()[0]]
                if last_wordlen_c_len == '<pad>': last_wordlen_c_len = 1
                start = index - last_wordlen_c_len
                last_word_emb.append(torch.mean(encode_out[idy][start:index], 0).unsqueeze(0))
            last_word_emb = torch.cat(last_word_emb, 0)

            z_in = torch.cat([last_word_emb, last_wordlen_emb, last_pos_emb], 1)
            ##### z: variable (batch_size, pos_dims+wordlen_dims+2*lstm_hiddens)
            z = self.dropout_lstm(F.tanh(self.fc(z_in)))
            ##### z: variable (batch_size, linear_hiddens)
        h, c = self.lstmcell(z, (h, c))
        return h, c

