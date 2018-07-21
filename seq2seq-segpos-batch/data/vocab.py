from data.alphabet import Alphabet
from collections import Counter
import data.utils as utils
import math
from data.instance import Instance
import data.instance as instance
unksymbol = '-unk-'
nullsymbol = '-NULL-'
sep = 'SEP'
app = 'APP'
pad = '<pad>'

class Data():
    def __init__(self):
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('char')
        self.bichar_alphabet = Alphabet('bichar')

        self.pos_alphabet = Alphabet('pos', is_label=True)
        self.char_type_alphabet = Alphabet('type', is_label=True)

        self.extchar_alphabet = Alphabet('extchar')
        self.extbichar_alphabet = Alphabet('extbichar')

        self.segpos_alphabet = Alphabet('segpos', is_label=True)
        self.wordlen_alphabet = Alphabet('wordlen')

        self.number_normalized = False
        self.norm_word_emb = False
        self.norm_char_emb = False
        self.norm_bichar_emb = False

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.pretrain_bichar_embedding = None

        self.wordPadID = 0
        self.charPadID = 0
        self.bicharPadID = 0
        self.charTypePadID = 0
        self.wordlenPadID = 0
        self.posPadID = 0
        self.appID = 0
        self.actionPadID = 0

        self.word_num = 0
        self.char_num = 0
        self.pos_num = 0
        self.bichar_num = 0
        self.segpos_num = 0
        self.wordlen_num = 0
        self.char_type_num = 0

        self.extchar_num = 0
        self.extbichar_num = 0

        self.wordlen_max = 7

    def build_alphabet(self, word_counter, char_counter, extchar_counter, bichar_counter, extbichar_counter, char_type_counter, pos_counter, wordlen_counter, gold_counter, shrink_feature_threshold):
        # 可以优化，但是会比较混乱
        for word, count in word_counter.most_common():
            # if count > shrink_feature_threshold:
            self.word_alphabet.add(word, count)
        for char, count in char_counter.most_common():
            if count > shrink_feature_threshold:
                self.char_alphabet.add(char, count)
        for extchar, count in extchar_counter.most_common():
            # if count > shrink_feature_threshold:
            self.extchar_alphabet.add(extchar, count)
        for bichar, count in bichar_counter.most_common():
            if count > shrink_feature_threshold:
                self.bichar_alphabet.add(bichar, count)
        for extbichar, count in extbichar_counter.most_common():
            # if count > shrink_feature_threshold:
            self.extbichar_alphabet.add(extbichar, count)
        for char_type, count in char_type_counter.most_common():
            # if count > shrink_feature_threshold:
            self.char_type_alphabet.add(char_type, count)
        for pos, count in pos_counter.most_common():
            # if count > shrink_feature_threshold:
            self.pos_alphabet.add(pos, count)
        for wordlen, count in wordlen_counter.most_common():
            # if count > shrink_feature_threshold:
            self.wordlen_alphabet.add(wordlen, count)
        for segpos, count in gold_counter.most_common():
            # if count > shrink_feature_threshold:
            self.segpos_alphabet.add(segpos, count)
        # another method
        # reverse = lambda x: dict(zip(x, range(len(x))))
        # self.word_alphabet.word2id = reverse(self.word_alphabet.id2word)
        # self.label_alphabet.word2id = reverse(self.label_alphabet.id2word)

        ##### check
        if len(self.word_alphabet.word2id) != len(self.word_alphabet.id2word) or len(self.word_alphabet.id2count) != len(self.word_alphabet.id2word):
            print('there are errors in building word alphabet.')
        if len(self.char_alphabet.word2id) != len(self.char_alphabet.id2word) or len(self.char_alphabet.id2count) != len(self.char_alphabet.id2word):
            print('there are errors in building char alphabet.')


    def fix_alphabet(self):
        self.word_num = self.word_alphabet.close()
        self.char_num = self.char_alphabet.close()
        self.pos_num = self.pos_alphabet.close()
        self.bichar_num = self.bichar_alphabet.close()
        self.segpos_num = self.segpos_alphabet.close()
        # self.wordlen_max = self.wordlen_alphabet.size()-2            ######
        # print(self.wordlen_max)
        # self.wordlen_alphabet.add(self.wordlen_max+1)
        self.wordlen_num = self.wordlen_alphabet.close()
        # print(self.wordlen_num)
        self.char_type_num = self.char_type_alphabet.close()

    def fix_static_alphabet(self):
        self.extchar_num = self.extchar_alphabet.close()
        self.extbichar_num = self.extbichar_alphabet.close()


    def get_instance(self, file, run_insts, shrink_feature_threshold):
        insts = []
        word_counter = Counter()
        char_counter = Counter()
        bichar_counter = Counter()
        char_type_counter = Counter()
        gold_counter = Counter()
        pos_counter = Counter()
        extchar_counter = Counter()
        extbichar_counter = Counter()
        wordlen_counter = Counter()
        count = 0
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if run_insts == count: break
                line = line.strip().split(' ')
                inst = Instance()
                char_num = 0
                start = 0
                for idx, ele in enumerate(line):
                    word = ele.split('_')[0]
                    if self.number_normalized: word = utils.normalize_word(word)
                    inst.words.append(word)
                    word_counter[word] += 1
                    pos = ele.split('_')[1]
                    inst.gold_pos.append(pos)
                    pos_counter[pos] += 1
                    word_list = list(word)
                    if len(word_list) > 6:
                        # inst.word_len.append(7)
                        cur_len = 7
                    else:
                        # inst.word_len.append(len(word_list))
                        cur_len = len(word_list)
                    inst.word_len.append(cur_len)
                    wordlen_counter[cur_len] += 1
                    for id, char in enumerate(word):
                        if idx == 0 and id == 0:
                            inst.last_wordlen.append('<pad>')
                            inst.last_pos.append('<pad>')
                        elif idx != 0 and id == 0:
                            last_wordlen_cur = len(inst.words[-2])
                            if last_wordlen_cur > 6: last_wordlen_cur = 7
                            inst.last_wordlen.append(last_wordlen_cur)   ### 因为当前的word和pos已经放进列表里了
                            inst.last_pos.append(inst.gold_pos[-2])
                        else:
                            last_wordlen_cur = id
                            if last_wordlen_cur > 6: last_wordlen_cur = 7
                            inst.last_wordlen.append(last_wordlen_cur)
                            inst.last_pos.append(inst.gold_pos[-1])
                        inst.chars.append(char)
                        inst.extchars.append(char)
                        char_counter[char] += 1
                        extchar_counter[char] += 1

                        char_type = instance.char_type(char)
                        # print(char_type)
                        inst.char_type.append(char_type)
                        char_type_counter[char_type] += 1
                        if id == 0:
                            inst.gold_action.append(sep+'#'+pos)
                            gold_counter[sep+'#'+pos] += 1
                            start = char_num
                        else:
                            inst.gold_action.append(app)
                            gold_counter[app] += 1
                        char_num += 1
                    inst.word_seg.append('['+str(start)+','+str(char_num)+']')
                    inst.word_seg_pos.append('['+str(start)+','+str(char_num)+']'+pos)
                right_bichars = []

                if len(inst.chars) == 1:
                    char = inst.chars[0]
                    inst.left_bichars.append(nullsymbol+char)
                    inst.extleft_bichars.append(nullsymbol+char)
                    right_bichars.append(char+nullsymbol)
                    bichar_counter[nullsymbol+char] += 1
                    bichar_counter[char+nullsymbol] += 1
                    extbichar_counter[nullsymbol+char] += 1
                    extbichar_counter[char+nullsymbol] += 1
                else:
                    for id, char in enumerate(inst.chars):
                        if id == 0:
                            inst.left_bichars.append(nullsymbol+char)
                            # inst.right_bichars.append(char+inst.chars[id+1])
                            inst.extleft_bichars.append(nullsymbol+char)
                            # inst.extright_bichars.append(char+inst.chars[id+1])
                            right_bichars.append(char+inst.chars[id+1])
                            # right_index_mark.append(1)
                            bichar_counter[nullsymbol+char] += 1
                            bichar_counter[char+inst.chars[id+1]] += 1
                            extbichar_counter[nullsymbol+char] += 1
                            extbichar_counter[char+inst.chars[id+1]] += 1
                        elif id == (len(inst.chars)-1):
                            inst.left_bichars.append(inst.chars[id-1]+char)
                            # inst.right_bichars.append(char+nullsymbol)
                            inst.extleft_bichars.append(inst.chars[id - 1] + char)
                            # inst.extright_bichars.append(char + nullsymbol)
                            right_bichars.append(char+nullsymbol)
                            # right_index_mark.append(1)
                            bichar_counter[inst.chars[id-1]+char] += 1
                            bichar_counter[char+nullsymbol] += 1
                            extbichar_counter[inst.chars[id-1]+char] += 1
                            extbichar_counter[char+nullsymbol] += 1
                        else:
                            inst.left_bichars.append(inst.chars[id-1]+char)
                            # inst.right_bichars.append(char+inst.chars[id+1])
                            inst.extleft_bichars.append(inst.chars[id - 1] + char)
                            # inst.extright_bichars.append(char + inst.chars[id + 1])
                            right_bichars.append(char+inst.chars[id+1])
                            # right_index_mark.append(1)
                            bichar_counter[inst.chars[id-1]+char] += 1
                            bichar_counter[char+inst.chars[id+1]] += 1
                            extbichar_counter[inst.chars[id-1]+char] += 1
                            extbichar_counter[char+inst.chars[id+1]] += 1
                # right_bichars = list(reversed(right_bichars))         #####
                # print(right_bichars)
                # print(right_index_mark)
                # right_index_mark = list(reversed(right_index_mark))
                inst.right_bichars = right_bichars
                inst.extright_bichars = right_bichars
                count += 1
                insts.append(inst)


        if not self.word_alphabet.fix_flag:
            self.build_alphabet(word_counter, char_counter, extchar_counter, bichar_counter, extbichar_counter, char_type_counter, pos_counter, wordlen_counter, gold_counter, shrink_feature_threshold)
        # insts_index = []

        for inst in insts:
            inst.words_index = [self.word_alphabet.get_index(w) for w in inst.words]
            inst.chars_index = [self.char_alphabet.get_index(c) for c in inst.chars]
            inst.extchars_index = [self.extchar_alphabet.get_index(ec) for ec in inst.extchars]
            inst.left_bichar_index = [self.bichar_alphabet.get_index(b) for b in inst.left_bichars]
            inst.right_bichar_index = [self.bichar_alphabet.get_index(b) for b in inst.right_bichars]
            inst.extleft_bichar_index = [self.extbichar_alphabet.get_index(eb) for eb in inst.extleft_bichars]
            inst.extright_bichar_index = [self.extbichar_alphabet.get_index(eb) for eb in inst.extright_bichars]
            inst.pos_index = [self.pos_alphabet.get_index(p) for p in inst.gold_pos]
            inst.char_type_index = [self.char_type_alphabet.get_index(t) for t in inst.char_type]
            # inst.char_type_index = []
            # for t in inst.char_type:
            #     print(t)
            #     print(self.char_type_alphabet.word2id)
            #     temp = self.char_type_alphabet.get_index(t)
            #     print(temp)
            #     inst.char_type_index.append(temp)
            inst.segpos_index = [self.segpos_alphabet.get_index(g) for g in inst.gold_action]
            inst.word_len_index = [self.wordlen_alphabet.get_index(w) for w in inst.word_len]
            inst.last_wordlen_index = [self.wordlen_alphabet.get_index(l) for l in inst.last_wordlen]
            inst.last_pos_index = [self.pos_alphabet.get_index(p) for p in inst.last_pos]

        self.wordPadID = self.word_alphabet.get_index(pad)
        self.charPadID = self.char_alphabet.get_index(pad)
        self.bicharPadID = self.bichar_alphabet.get_index(pad)
        self.charTypePadID = self.char_type_alphabet.get_index(pad)
        self.wordlenPadID = self.wordlen_alphabet.get_index(pad)
        self.posPadID = self.pos_alphabet.get_index(pad)
        self.appID = self.segpos_alphabet.get_index(app)
        self.actionPadID = self.segpos_alphabet.get_index(pad)

        ##### sorted sentences
        # insts_sorted, insts_index_sorted = utils.sorted_instances(insts, insts_index)
        return insts

    def build_word_pretrain_emb(self, emb_path, word_dims):
        self.pretrain_word_embedding = utils.load_pretrained_emb_uniform(emb_path, self.word_alphabet.word2id, word_dims, self.norm_word_emb)

    # 可以优化
    def build_char_pretrain_emb(self, emb_path, char_dims):
        self.pretrain_char_embedding = utils.load_pretrained_emb_uniform(emb_path, self.extchar_alphabet.word2id, char_dims, self.norm_char_emb)
    # 可以优化
    def build_bichar_pretrain_emb(self, emb_path, bichar_dims):
        self.pretrain_bichar_embedding = utils.load_pretrained_emb_uniform(emb_path, self.extbichar_alphabet.word2id, bichar_dims, self.norm_bichar_emb)

    def generate_batch_buckets(self, batch_size, insts):
        # insts_length = list(map(lambda t: len(t) + 1, inst[0] for inst in insts))
        # insts_length = list(len(inst[0]+1) for inst in insts)
        # if len(insts) % batch_size == 0:
        #     batch_num = len(insts) // batch_size
        # else:
        #     batch_num = len(insts) // batch_size + 1
        batch_num = int(math.ceil(len(insts) / batch_size))

        buckets = [[[], [], [],[],[],[],[],[],[],[],[],[],[],[],[]] for _ in range(batch_num)]
        # labels_raw = [[] for _ in range(batch_num)]
        inst_save = []
        for id, inst in enumerate(insts):
            idx = id // batch_size
            if id == 0 or id % batch_size != 0:
                inst_save.append(inst)
            elif id % batch_size == 0:
                assert len(inst_save) == batch_size
                inst_sorted = utils.sort_instances(inst_save)
                max_length = len(inst_sorted[0].chars_index)
                for idy in range(batch_size):
                    cur_length = len(inst_sorted[idy].chars_index)

                    buckets[idx-1][0].append(inst_sorted[idy].chars_index + [self.char_alphabet.word2id['<pad>']] * (max_length - cur_length))
                    buckets[idx-1][1].append(inst_sorted[idy].left_bichar_index + [self.bichar_alphabet.word2id['<pad>']] * (max_length - cur_length))
                    buckets[idx-1][2].append(inst_sorted[idy].right_bichar_index + [self.bichar_alphabet.word2id['<pad>']]*(max_length-cur_length))
                    buckets[idx-1][3].append(inst_sorted[idy].extchars_index + [self.extchar_alphabet.word2id['<pad>']]*(max_length-cur_length))
                    buckets[idx-1][4].append(inst_sorted[idy].extleft_bichar_index + [self.extbichar_alphabet.word2id['<pad>']]*(max_length-cur_length))
                    buckets[idx-1][5].append(inst_sorted[idy].extright_bichar_index + [self.extbichar_alphabet.word2id['<pad>']]*(max_length-cur_length))
                    buckets[idx-1][6].append(inst_sorted[idy].char_type_index + [self.char_type_alphabet.word2id['<pad>']]*(max_length-cur_length))
                    buckets[idx-1][7].append(inst_sorted[idy].segpos_index + [self.segpos_alphabet.word2id['<pad>']]*(max_length-cur_length))
                    buckets[idx-1][8].append(inst_sorted[idy].last_wordlen_index + [self.wordlen_alphabet.word2id['<pad>']]*(max_length-cur_length))
                    buckets[idx-1][9].append(inst_sorted[idy].last_pos_index + [self.pos_alphabet.word2id['<pad>']]*(max_length-cur_length))
                    buckets[idx-1][10].append(inst_sorted[idy].gold_action + ['<pad>']*(max_length-cur_length))
                    buckets[idx-1][11].append(inst_sorted[idy].gold_pos + ['<pad>']*(max_length-cur_length))
                    buckets[idx-1][12].append(inst_sorted[idy].chars + ['<pad>']*(max_length-cur_length))
                    buckets[idx-1][13].append(inst_sorted[idy].words + ['<pad>']*(max_length-cur_length))

                    buckets[idx - 1][-1].append([1] * cur_length + [0] * (max_length - cur_length))
                    # labels_raw[idx-1].append(inst_sorted[idy][-1])
                inst_save = []
                inst_save.append(inst)
        if inst_save != []:
            inst_sorted = utils.sort_instances(inst_save)
            max_length = len(inst_sorted[0].chars_index)
            for idy in range(len(inst_sorted)):
                cur_length = len(inst_sorted[idy].chars_index)
                buckets[batch_num-1][0].append(inst_sorted[idy].chars_index + [self.char_alphabet.word2id['<pad>']] * (max_length - cur_length))
                buckets[batch_num-1][1].append(inst_sorted[idy].left_bichar_index + [self.bichar_alphabet.word2id['<pad>']] * (max_length - cur_length))
                buckets[batch_num-1][2].append(inst_sorted[idy].right_bichar_index + [self.bichar_alphabet.word2id['<pad>']] * (max_length - cur_length))
                buckets[batch_num-1][3].append(inst_sorted[idy].extchars_index + [self.extchar_alphabet.word2id['<pad>']] * (max_length - cur_length))
                buckets[batch_num-1][4].append(inst_sorted[idy].extleft_bichar_index + [self.extbichar_alphabet.word2id['<pad>']] * (max_length - cur_length))
                buckets[batch_num-1][5].append(inst_sorted[idy].extright_bichar_index + [self.extbichar_alphabet.word2id['<pad>']] * (max_length - cur_length))
                buckets[batch_num-1][6].append(inst_sorted[idy].char_type_index + [self.char_type_alphabet.word2id['<pad>']] * (max_length - cur_length))
                buckets[batch_num-1][7].append(inst_sorted[idy].segpos_index + [self.segpos_alphabet.word2id['<pad>']]*(max_length-cur_length))
                buckets[batch_num-1][8].append(inst_sorted[idy].last_wordlen_index + [self.wordlen_alphabet.word2id['<pad>']]*(max_length-cur_length))
                buckets[batch_num-1][9].append(inst_sorted[idy].last_pos_index + [self.pos_alphabet.word2id['<pad>']]*(max_length-cur_length))
                buckets[batch_num-1][10].append(inst_sorted[idy].gold_action + ['<pad>']*(max_length-cur_length))
                buckets[batch_num-1][11].append(inst_sorted[idy].gold_pos + ['<pad>']*(max_length-cur_length))
                buckets[batch_num-1][12].append(inst_sorted[idy].chars + ['<pad>']*(max_length-cur_length))
                buckets[batch_num-1][13].append(inst_sorted[idy].words + ['<pad>']*(max_length-cur_length))

                buckets[batch_num-1][-1].append([1] * cur_length + [0] * (max_length - cur_length))
                # labels_raw[batch_num-1].append(inst_sorted[idy][-1])
        return buckets

    def generate_batch_buckets_save(self, batch_size, insts, char=False):
        # insts_length = list(map(lambda t: len(t) + 1, inst[0] for inst in insts))
        # insts_length = list(len(inst[0]+1) for inst in insts)
        # if len(insts) % batch_size == 0:
        #     batch_num = len(insts) // batch_size
        # else:
        #     batch_num = len(insts) // batch_size + 1
        batch_num = int(math.ceil(len(insts) / batch_size))

        if char:
            buckets = [[[], [], [], []] for _ in range(batch_num)]
        else:
            buckets = [[[], [], []] for _ in range(batch_num)]
        max_length = 0
        for id, inst in enumerate(insts):
            idx = id // batch_size
            if id % batch_size == 0:
                max_length = len(inst[0]) + 1
            cur_length = len(inst[0])

            buckets[idx][0].append(inst[0] + [self.word_alphabet.word2id['<pad>']] * (max_length - cur_length))
            buckets[idx][1].append([self.label_alphabet.word2id['<start>']] + inst[-1] + [self.label_alphabet.word2id['<pad>']] * (max_length - cur_length - 1))
            if char:
                char_length = len(inst[1][0])
                buckets[idx][2].append((inst[1] + [[self.char_alphabet.word2id['<pad>']] * char_length] * (max_length - cur_length)))
            buckets[idx][-1].append([1] * (cur_length + 1) + [0] * (max_length - (cur_length + 1)))

        return buckets











