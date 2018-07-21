import argparse
import data.config as config
from data.vocab import Data
import model.encode as encode
import model.decode as decode
import model.decode_batch as decode_batch
import model.decode_batch_opt as decode_batch_opt
import train
import torch
import random
import numpy as np
import time


if __name__ == '__main__':
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(666)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config-file', default='/data/disk1/song/CWS/seq2seq-segpos-complete-new-batch/examples/config.cfg')
    argparser.add_argument('--use-cuda', default=True)
    argparser.add_argument('--metric', default='exact', help='choose from [exact, binary, proportional]')

    # args = argparser.parse_known_args()
    args = argparser.parse_args()
    config = config.Configurable(args.config_file)

    data = Data()
    data.number_normalized = False
    data.use_cuda = args.use_cuda
    data.metric = args.metric

    test_time = time.time()
    train_insts = data.get_instance(config.train_file, config.run_insts,config.shrink_feature_thresholds)
    print('test getting train_insts time: ', time.time()-test_time)

    data.fix_alphabet()
    dev_insts = data.get_instance(config.dev_file, config.run_insts, config.shrink_feature_thresholds)
    print('test getting dev_insts time: ', time.time() - test_time)

    test_insts = data.get_instance(config.test_file, config.run_insts,config.shrink_feature_thresholds)
    ##### 可将整个emb词表放入，不专针对此份测试集
    data.fix_static_alphabet()
    print('test getting test_insts time: ', time.time() - test_time)

    # train_buckets, train_labels_raw = data.generate_batch_buckets(config.train_batch_size, train_insts_index, char=args.add_char)
    # dev_buckets, dev_labels_raw = data.generate_batch_buckets(len(dev_insts), dev_insts_index, char=args.add_char)
    # test_buckets, test_labels_raw = data.generate_batch_buckets(len(test_insts), test_insts_index, char=args.add_char)

    # print('test getting batch_insts time: ', time.time() - test_time)

    if config.pretrained_wordEmb_file != '':
        data.norm_word_emb = False
        data.build_word_pretrain_emb(config.pretrained_wordEmb_file, config.word_dims)
    if config.pretrained_charEmb_file != '':
        data.norm_char_emb = False
        data.build_char_pretrain_emb(config.pretrained_charEmb_file, config.char_dims)
    if config.pretrained_bicharEmb_file != '':
        data.norm_bichar_emb = False
        data.build_bichar_pretrain_emb(config.pretrained_bicharEmb_file, config.bichar_dims)

    encode = encode.Encode(config, data)
    decode = decode_batch.Decode(config, data)
    # decode = decode_batch_opt.Decode(config, data)
    if data.use_cuda:
        encode = encode.cuda()
        decode = decode.cuda()
    print('test building model time: ', time.time() - test_time)

    train.train_segpos(train_insts, dev_insts, test_insts, encode, decode, config, data)






