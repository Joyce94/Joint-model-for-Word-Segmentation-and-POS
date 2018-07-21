import torch
import torch.nn as nn
import time
import random
import numpy as np
import data.utils as utils
import model.crf
import model.evaluation_joint as evaluation_joint
import data.vocab as vocab
import torch.nn.functional as F
from model.eval import Eval

def to_scalar(vec):
    return vec.view(-1).data.tolist()[0]

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_segpos(train_insts, dev_insts, test_insts, encode, decode, config, params):
    print('training...')
    parameters_en = filter(lambda p: p.requires_grad, encode.parameters())
    # optimizer_en = torch.optim.SGD(params=parameters_en, lr=config.learning_rate, momentum=0.9, weight_decay=config.decay)
    optimizer_en = torch.optim.Adam(params=parameters_en, lr=config.learning_rate, weight_decay=config.decay)
    parameters_de = filter(lambda p: p.requires_grad, decode.parameters())
    # optimizer_de = torch.optim.SGD(params= parameters_de, lr=config.learning_rate, momentum=0.9, weight_decay=config.decay)
    optimizer_de = torch.optim.Adam(params=parameters_de, lr=config.learning_rate, weight_decay=config.decay)

    best_dev_f1_seg = float('-inf')
    best_dev_f1_pos = float('-inf')
    best_test_f1_seg = float('-inf')
    best_test_f1_pos = float('-inf')

    dev_eval_seg = Eval()
    dev_eval_pos = Eval()
    test_eval_seg = Eval()
    test_eval_pos = Eval()
    for epoch in range(config.maxIters):
        start_time = time.time()
        encode.train()
        decode.train()
        train_insts = utils.random_instances(train_insts)
        epoch_loss = 0
        train_buckets = params.generate_batch_buckets(config.train_batch_size, train_insts)

        for index in range(len(train_buckets)):
            batch_length = np.array([np.sum(mask) for mask in train_buckets[index][-1]])

            var_b, list_b, mask_v, length_v, gold_v = utils.patch_var(train_buckets[index], batch_length.tolist(), params)
            encode.zero_grad()
            decode.zero_grad()
            if mask_v.size(0) != config.train_batch_size:
                encode.hidden = encode.init_hidden(mask_v.size(0))
            else:
                encode.hidden = encode.init_hidden(config.train_batch_size)
            lstm_out = encode.forward(var_b, list_b, mask_v, batch_length.tolist())
            output, state = decode.forward(lstm_out, var_b, list_b, mask_v, batch_length.tolist(), is_train=True)
            #### output: variable (batch_size, max_length, segpos_num)

            # num_total = output.size(0)*output.size(1)
            # output = output.contiguous().view(num_total, output.size(2))
            # print(output)
            # gold_v = gold_v.view(num_total)
            # print(output)
            gold_v = gold_v.view(output.size(0))
            # print(gold_v)
            loss = F.cross_entropy(output, gold_v)
            loss.backward()

            nn.utils.clip_grad_norm(parameters_en, max_norm=config.clip_grad)
            nn.utils.clip_grad_norm(parameters_de, max_norm=config.clip_grad)

            optimizer_en.step()
            optimizer_de.step()
            epoch_loss += utils.to_scalar(loss)
        print('\nepoch is {}, average loss is {} '.format(epoch, (epoch_loss / (config.train_batch_size * len(train_buckets)))))
        # update lr
        # adjust_learning_rate(optimizer, config.learning_rate / (1 + (epoch + 1) * config.decay))
        # acc = float(correct_num) / float(gold_num)
        # print('\nepoch is {}, accuracy is {}'.format(epoch, acc))
        print('the {} epoch training costs time: {} s '.format(epoch, time.time()-start_time))

        print('\nDev...')
        dev_eval_seg.clear()
        dev_eval_pos.clear()
        test_eval_seg.clear()
        test_eval_pos.clear()

        start_time = time.time()
        dev_f1_seg, dev_f1_pos = eval_batch(dev_insts, encode, decode, config, params, test_eval_seg, test_eval_pos)
        print('the {} epoch dev costs time: {} s'.format(epoch, time.time()-start_time))

        if dev_f1_seg > best_dev_f1_seg:
            best_dev_f1_seg = dev_f1_seg
            if dev_f1_pos > best_dev_f1_pos: best_dev_f1_pos = dev_f1_pos
            print('\nTest...')
            start_time = time.time()
            test_f1_seg, test_f1_pos = eval_batch(test_insts, encode, decode, config, params, test_eval_seg, test_eval_pos)
            print('the {} epoch testing costs time: {} s'.format(epoch, time.time() - start_time))

            if test_f1_seg > best_test_f1_seg:
                best_test_f1_seg = test_f1_seg
            if test_f1_pos > best_test_f1_pos:
                best_test_f1_pos = test_f1_pos
            print('now, test fscore of seg is {}, test fscore of pos is {}, best test fscore of seg is {}, best fscore of pos is {} '.format(test_f1_seg, test_f1_pos, best_test_f1_seg, best_test_f1_pos))
            torch.save(encode.state_dict(), config.save_encode_path)
            torch.save(decode.state_dict(), config.save_decode_path)
        else:
            if dev_f1_pos > best_dev_f1_pos:
                best_dev_f1_pos = dev_f1_pos
                print('\nTest...')
                start_time = time.time()
                test_f1_seg, test_f1_pos = eval_batch(test_insts, encode, decode, config, params, test_eval_seg, test_eval_pos)
                print('the {} epoch testing costs time: {} s'.format(epoch, time.time() - start_time))

                if test_f1_seg > best_test_f1_seg:
                    best_test_f1_seg = test_f1_seg
                if test_f1_pos > best_test_f1_pos:
                    best_test_f1_pos = test_f1_pos
                print('now, test fscore of seg is {}, test fscore of pos is {}, best test fscore of seg is {}, best fscore of pos is {} '.format(test_f1_seg, test_f1_pos, best_test_f1_seg, best_test_f1_pos))
                torch.save(encode.state_dict(), config.save_encode_path)
                torch.save(decode.state_dict(), config.save_decode_path)
        print('now, dev fscore of seg is {}, dev fscore of pos is {}, best dev fscore of seg is {}, best dev fscore of pos is {}, best test fscore of seg is {}, best test fscore of pos is {}'.format(dev_f1_seg, dev_f1_pos, best_dev_f1_seg, best_test_f1_pos, best_test_f1_seg, best_test_f1_pos))


def eval(insts, encode, decode, config, params):
    encode.eval()
    decode.eval()
    insts = utils.random_instances(insts)
    buckets = params.generate_batch_buckets(len(insts), insts)

    batch_length = np.array([np.sum(mask) for mask in buckets[0][-1]])
    var_b, list_b, mask_v, length_v, gold_v = utils.patch_var(buckets[0], batch_length.tolist(), params)
    encode.zero_grad()
    decode.zero_grad()
    if mask_v.size(0) != config.test_batch_size:
        encode.hidden = encode.init_hidden(mask_v.size(0))
    else:
        encode.hidden = encode.init_hidden(config.test_batch_size)
    lstm_out = encode.forward(var_b, list_b, mask_v, batch_length.tolist())
    output, action = decode.forward(lstm_out, var_b, list_b, mask_v, batch_length.tolist(), is_train=False)
    ##### output: variable (batch_size, max_length, segpos_num)

    gold_index = list_b[1]

    f_score = evaluation_joint.eval_entity(gold_index, action, params)
    return f_score

def eval_batch(insts, encode, decode, config, params, eval_seg, eval_pos):
    encode.eval()
    decode.eval()
    insts = utils.random_instances(insts)
    buckets = params.generate_batch_buckets(config.test_batch_size, insts)

    gold_total = []
    pos_total = []
    word_total = []
    action_total = []
    eval_gold = []
    eval_poslabels = []
    for index in range(len(buckets)):
        batch_length = np.array([np.sum(mask) for mask in buckets[index][-1]])
        var_b, list_b, mask_v, length_v, gold_v = utils.patch_var(buckets[index], batch_length.tolist(), params)
        encode.zero_grad()
        decode.zero_grad()
        if mask_v.size(0) != config.test_batch_size:
            encode.hidden = encode.init_hidden(mask_v.size(0))
        else:
            encode.hidden = encode.init_hidden(config.test_batch_size)
        lstm_out = encode.forward(var_b, list_b, mask_v, batch_length.tolist())
        output, state = decode.forward(lstm_out, var_b, list_b, mask_v, batch_length.tolist(), is_train=False)

        gold_index = list_b[1]
        pos_index = list_b[2]
        word_index = list_b[-1]

        gold_total.extend(gold_index)
        pos_total.extend(pos_index)
        word_total.extend(word_index)

        for id in range(mask_v.size(0)):
            eval_gold.append(state[id].words)
            eval_poslabels.append(state[id].pos_labels)

    for idx in range(len(gold_total)):
        eval_seg, eval_pos = jointPRF(word_total[idx], pos_total[idx], eval_gold[idx], eval_poslabels[idx], eval_seg, eval_pos)
    p, r, f = eval_seg.getFscore()
    fscore_seg = f
    print('seg eval: precision = ', str(p), '%, recall = ', str(r), '%, f-score = ', str(f), "%")
    p, r, f = eval_pos.getFscore()
    fscore_pos = f
    print('pos eval: precision = ', str(p), '%, recall = ', str(r), '%, f-score = ', str(f), "%")

    # fscore_seg, fscore_pos = evaluation_joint.eval_entity(gold_total, action_total, params)
    return fscore_seg, fscore_pos

def jointPRF(gold_seg, gold_pos, words, posLabels, seg_eval, pos_eval):
    predict_seg = []
    predict_pos = []
    origin_seg = []
    origin_pos = []

    count = 0
    for idx in range(len(words)):
        w = words[idx]
        if w == '<pad>': break
        posLabel = posLabels[idx]
        predict_seg.append('[' + str(count) + ',' + str(count + len(w)) + ']')
        predict_pos.append('[' + str(count) + ',' + str(count + len(w)) + ']' + posLabel)
        # print('[' + str(count) + ',' + str(count + len(w)) + ']')
        # print('[' + str(count) + ',' + str(count + len(w)) + ']' + posLabel)
        count += len(w)
    count = 0
    for idx in range(len(gold_seg)):
        w = gold_seg[idx]
        if w == '<pad>': break
        posLabel = gold_pos[idx]
        origin_seg.append('[' + str(count) + ',' + str(count + len(w)) + ']')
        origin_pos.append('[' + str(count) + ',' + str(count + len(w)) + ']' + posLabel)
        # print('[' + str(count) + ',' + str(count + len(w)) + ']')
        # print('[' + str(count) + ',' + str(count + len(w)) + ']' + posLabel)
        count += len(w)

    seg_eval.gold_num += len(origin_seg)
    seg_eval.predict_num += len(predict_seg)
    for p in predict_seg:
        if p in origin_seg:
            seg_eval.correct_num += 1

    pos_eval.gold_num += len(origin_pos)
    pos_eval.predict_num += len(predict_pos)
    for p in predict_pos:
        if p in origin_pos:
            pos_eval.correct_num += 1
    return seg_eval, pos_eval