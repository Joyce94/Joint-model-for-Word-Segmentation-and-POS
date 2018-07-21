import torch
import torch.nn as nn
import time
import random
import numpy as np
import data.utils as utils
import model.evaluation_joint as evaluation_joint
import data.vocab as vocab
import torch.nn.functional as F

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

            ##### method 1: decode for batch
            output, state = decode.forward(lstm_out, var_b, list_b, mask_v, batch_length.tolist(), is_train=True)
            ##### method 2: decode using optimize
            # output, state = decode.forward_train(lstm_out, var_b, list_b, mask_v, batch_length.tolist())
            #### output: variable (batch_size, max_length, segpos_num)
            # gold = list_b[1]

            num_total = output.size(0)*output.size(1)
            output = output.contiguous().view(num_total, output.size(2))
            # print(output)
            gold_v = gold_v.view(num_total)
            # print(output)
            # gold_v = gold_v.view(output.size(0))
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

        start_time = time.time()
        dev_f1_seg, dev_f1_pos = eval_batch(dev_insts, encode, decode, config, params)
        print('the {} epoch dev costs time: {} s'.format(epoch, time.time()-start_time))

        if dev_f1_seg > best_dev_f1_seg:
            best_dev_f1_seg = dev_f1_seg
            if dev_f1_pos > best_dev_f1_pos: best_dev_f1_pos = dev_f1_pos
            print('\nTest...')
            start_time = time.time()
            test_f1_seg, test_f1_pos = eval_batch(test_insts, encode, decode, config, params)
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
                test_f1_seg, test_f1_pos = eval_batch(test_insts, encode, decode, config, params)
                print('the {} epoch testing costs time: {} s'.format(epoch, time.time() - start_time))

                if test_f1_seg > best_test_f1_seg:
                    best_test_f1_seg = test_f1_seg
                if test_f1_pos > best_test_f1_pos:
                    best_test_f1_pos = test_f1_pos
                print('now, test fscore of seg is {}, test fscore of pos is {}, best test fscore of seg is {}, best fscore of pos is {} '.format(test_f1_seg, test_f1_pos, best_test_f1_seg, best_test_f1_pos))
                torch.save(encode.state_dict(), config.save_encode_path)
                torch.save(decode.state_dict(), config.save_decode_path)
        print('now, dev fscore of seg is {}, dev fscore of pos is {}, best dev fscore of seg is {}, best dev fscore of pos is {}, best test fscore of seg is {}, best test fscore of pos is {}'.format(dev_f1_seg, dev_f1_pos, best_dev_f1_seg, best_dev_f1_pos, best_test_f1_seg, best_test_f1_pos))


def eval_batch(insts, encode, decode, config, params):
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
        output, state = decode.forward(lstm_out, var_b, list_b, mask_v, batch_length.tolist())

        gold_index = list_b[1]
        pos_index = list_b[2]
        word_index = list_b[-1]

        gold_total.extend(gold_index)
        pos_total.extend(pos_index)
        word_total.extend(word_index)

        action_total.extend(state.action)
        eval_gold.extend(state.words_record)
        eval_poslabels.extend(state.pos_record)

    fscore_seg, fscore_pos = evaluation_joint.eval_entity(gold_total, action_total, params)
    return fscore_seg, fscore_pos

