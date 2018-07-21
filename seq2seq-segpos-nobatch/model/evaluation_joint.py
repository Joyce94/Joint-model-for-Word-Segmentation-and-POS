class Entity():
    def __init__(self, start, end, category, is_pos=False):
        super(Entity, self).__init__()
        self.start = start
        self.end = end
        self.category = category

    def equal(self, entity):
        return self.start == entity.start and self.end == entity.end and self.category == entity.category

    def match(self, entity):
        span = set(range(int(self.start), int(self.end) + 1))
        entity_span = set(range(int(entity.start), int(entity.end) + 1))
        return len(span.intersection(entity_span)) and self.category == entity.category

    def propor_score(self, entity):
        span = set(range(int(self.start), int(self.end) + 1))
        entity_span = set(range(int(entity.start), int(entity.end) + 1))
        return float(len(span.intersection(entity_span))) / float(len(span))


def Extract_entity(labels, category_set):
    idx = 0
    ent_seg = []
    ent_pos = []
    # print(labels)       # ['SEP#NR', 'APP', 'APP', 'SEP#AD', 'APP', 'SEP#VV'
    # print(len(labels))  # 163
    while (idx < len(labels)):
        index = labels[idx].find('#')
        if index != -1:
            idy = idx
            endpos = -1
            while (idy < len(labels)):
                cur = labels[idy].find('#')
                if cur == -1:
                    endpos = idy - 1
                    break
                endpos = idy
                idy += 1
            category1 = 'SEG'
            category2 = 'POS'
            entity1 = Entity(idx, endpos, category1)
            ent_seg.append(entity1)
            entity2 = Entity(idx, endpos, category2)
            ent_pos.append(entity2)
            idx = endpos
        idx += 1
    category_num = len(category_set)
    category_list = [e for e in category_set]
    # print(category_set)       # ('SEG', 'POS')
    # print(category_list)      # ['SEG', 'POS']
    # print(category_num)       # 2
    entity_group = []
    for i in range(category_num):
        entity_group.append([])
    # print(entity_group)       # [[], []]
    for entity in ent_seg:
        if entity.category == 'SEG':
            entity_group[0].append(entity)
    for entity in ent_pos:
        if entity.category == 'POS':
            entity_group[1].append(entity)
    return entity_group


def Extract_action_old(action, category_set):
    idx = 0
    ent_seg = []
    ent_pos = []
    # print(labels.action)
    while (idx < len(action)):
        index = action[idx].find('#')
        if index != -1:
            idy = idx
            endpos = -1
            while (idy < len(action)):
                cur = action[idy].find('#')
                if cur == -1:
                    endpos = idy - 1
                    break
                endpos = idy
                idy += 1
            category1 = 'SEG'
            category2 = 'POS'
            entity1 = Entity(idx, endpos, category1)
            ent_seg.append(entity1)
            entity2 = Entity(idx, endpos, category2)
            ent_pos.append(entity2)
            idx = endpos
        idx += 1
    category_num = len(category_set)
    category_list = [e for e in category_set]
    # print(category_set)       # ('SEG', 'POS')
    # print(category_list)      # ['SEG', 'POS']
    # print(category_num)       # 2
    entity_group = []
    for i in range(category_num):
        entity_group.append([])
    # print(entity_group)       # [[], []]
    for entity in ent_seg:
        if entity.category == 'SEG':
            entity_group[0].append(entity)
    for entity in ent_pos:
        if entity.category == 'POS':
            entity_group[1].append(entity)
    return entity_group


def Extract_segpos(labels, category_set):
    ent_seg = []
    ent_pos = []
    # print(labels)                 # ['SEP#NR', 'APP', 'APP', 'SEP#AD', 'APP', 'SEP#VV'
    start = 0
    end = -1
    pos = ''
    for id, ele in enumerate(labels):
        if ele == '<pad>':
            end = id - 1
            # print('[' + str(start) + ',' + str(end) + ']')
            # print('[' + str(start) + ',' + str(end) + ']' + pos)
            category1 = 'SEG'
            # category2 = 'POS'
            entity1 = Entity(start, end, category1)
            ent_seg.append(entity1)
            entity2 = Entity(start, end, pos, is_pos=True)
            ent_pos.append(entity2)
            break
        index = ele.find('#')
        if index != -1:
            if pos != '':
                end = id - 1
                # print('[' + str(start) + ',' + str(end) + ']')
                # print('[' + str(start) + ',' + str(end) + ']' + pos)
                category1 = 'SEG'
                # category2 = 'POS'
                entity1 = Entity(start, end, category1)
                ent_seg.append(entity1)
                entity2 = Entity(start, end, pos, is_pos=True)
                ent_pos.append(entity2)
            start = id
            pos = ele.split('#')[1]
        if id == len(labels)-1:
            end = id
            category1 = 'SEG'
            entity1 = Entity(start, end, category1)
            ent_seg.append(entity1)
            entity2 = Entity(start, end, pos, is_pos=True)
            ent_pos.append(entity2)
    category_num = len(category_set)
    category_list = [e for e in category_set]
    # print(category_set)       # ('SEG', 'POS')
    # print(category_list)      # ['SEG', 'POS']
    # print(category_num)       # 2
    entity_group = []
    for i in range(category_num):
        entity_group.append([])
    # print(entity_group)       # [[], []]
    for entity in ent_seg:
        if entity.category == 'SEG':
            entity_group[0].append(entity)
    for entity in ent_pos:
        # if entity.category == 'POS':
        entity_group[1].append(entity)
    return entity_group


def Extract_action(action, category_set):
    ent_seg = []
    ent_pos = []
    # print(labels)                 # ['SEP#NR', 'APP', 'APP', 'SEP#AD', 'APP', 'SEP#VV'
    start = 0
    end = -1
    pos = ''
    for id, ele in enumerate(action):
        if ele == '<pad>':
            end = id - 1
            category1 = 'SEG'
            entity1 = Entity(start, end, category1)
            ent_seg.append(entity1)
            entity2 = Entity(start, end, pos, is_pos=True)
            ent_pos.append(entity2)
            break
        index = ele.find('#')
        if index != -1:     # 遇到下一个#，就把前一个entity加进去
            if pos != '':
                end = id - 1
                category1 = 'SEG'
                entity1 = Entity(start, end, category1)
                ent_seg.append(entity1)
                entity2 = Entity(start, end, pos, is_pos=True)
                ent_pos.append(entity2)
            start = id
            pos = ele.split('#')[1]
        if id == len(action)-1:   # 如果句子中没有pad, 则计算到最后的时候要把最后那个加进去。最后一个会有两种情况。#/app
            end = id
            category1 = 'SEG'
            entity1 = Entity(start, end, category1)
            ent_seg.append(entity1)
            entity2 = Entity(start, end, pos, is_pos=True)
            ent_pos.append(entity2)
    category_num = len(category_set)
    category_list = [e for e in category_set]
    # print(category_set)       # ('SEG', 'POS')
    # print(category_list)      # ['SEG', 'POS']
    # print(category_num)       # 2
    entity_group = []
    for i in range(category_num):
        entity_group.append([])
    # print(entity_group)       # [[], []]
    for entity in ent_seg:
        if entity.category == 'SEG':
            entity_group[0].append(entity)
    for entity in ent_pos:
        # if entity.category == 'POS':
        entity_group[1].append(entity)
    return entity_group


class Eval():
    def __init__(self, category_set, dataset_num):
        self.category_set = category_set
        self.dataset_sum = dataset_num

        self.precision_c = []
        self.recall_c = []
        self.f1_score_c = []

    def clear(self):
        self.real_num = 0
        self.predict_num = 0
        self.correct_num = 0
        self.correct_num_p = 0

    def set_eval_var(self):
        category_num = len(self.category_set)
        self.B = []
        b = list(range(4))
        for i in range(category_num + 1):
            bb = [0 for e in b]
            self.B.append(bb)

    def Exact_match(self, predict_set, gold_set):
        self.clear()
        self.gold_num = len(gold_set)
        self.predict_num = len(predict_set)
        # correct_num = 0
        for p in predict_set:
            for g in gold_set:
                if p.equal(g):
                    self.correct_num += 1
                    break
        result = (self.gold_num, self.predict_num, self.correct_num)
        return result

    def Binary_evaluate(self, predict_set, gold_set):
        self.clear()
        self.gold_num = len(gold_set)
        self.predict_num = len(predict_set)
        for p in predict_set:
            for g in gold_set:
                if p.match(g):
                    self.correct_num_p += 1
                    break
        for g in gold_set:
            for p in predict_set:
                if g.match(p):
                    self.correct_num += 1
                    break
        result = (self.gold_num, self.predict_num, self.correct_num, self.correct_num_p)
        return result

    def Propor_evaluate(self, predict_set, gold_set):
        self.clear()
        self.gold_num = len(gold_set)
        self.predict_num = len(predict_set)
        for p in predict_set:
            for g in gold_set:
                if p.match(g):
                    self.correct_num_p += p.propor_score(g)
                    break
        for g in gold_set:
            for p in predict_set:
                if g.match(p):
                    self.correct_num += g.propor_score(p)
                    break
        result = (self.gold_num, self.predict_num, self.correct_num, self.correct_num_p)
        return result

    def calc_f1_score(self, eval_type):
        category_list = [e for e in self.category_set]
        category_num = len(self.category_set)
        if eval_type == 'exact':
            for iter in range(category_num):
                result = self.get_f1_score_e(self.B[iter + 1][0], self.B[iter + 1][1], self.B[iter + 1][2])
                self.precision_c.append(result[0])
                self.recall_c.append(result[1])
                self.f1_score_c.append(result[2])

        print('\n(The total number of dataset: {})\n'.format(self.dataset_sum))
        print('\rEvalution')
        for index in range(category_num):
            print('\r   {} - precision: {:.4f}% recall: {:.4f}% f1_score: {:.4f} ({}/{}/{})'.format(
                category_list[index], (self.precision_c[index] * 100), (self.recall_c[index] * 100),
                (self.f1_score_c[index]*100), self.B[index + 1][2], self.B[index + 1][1], self.B[index + 1][0]))
        return self.f1_score_c[0], self.f1_score_c[1]

    def overall_evaluate(self, predict_set, gold_set, eval_type):
        if eval_type == 'exact':
            return self.Exact_match(predict_set, gold_set)

    def eval(self, gold_labels, predict_labels, eval_type):
        for index in range(len(gold_labels)):
            gold_entity_group = Extract_segpos(gold_labels[index], self.category_set)
            pre_entity_group = Extract_action(predict_labels[index], self.category_set)

            for iter in range(len(self.category_set)):
                result = self.overall_evaluate(pre_entity_group[iter], gold_entity_group[iter], eval_type)
                for i in range(len(result)):
                    self.B[iter + 1][i] += result[i]

    def get_f1_score_e(self, real_num, predict_num, correct_num):
        if predict_num != 0:
            precision = correct_num / predict_num
        else:
            precision = 0.0
        if real_num != 0:
            recall = correct_num / real_num
        else:
            recall = 0.0
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        result = (precision, recall, f1_score)
        return result

    def get_f1_score(self, real_num, predict_num, correct_num_r, correct_num_p):
        if predict_num != 0:
            precision = correct_num_p / predict_num
        else:
            precision = 0.0
        if real_num != 0:
            recall = correct_num_r / real_num
        else:
            recall = 0.0
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        result = (precision, recall, f1_score)
        return result


def eval_entity(gold_labels, predict_labels, params):
    # dataset_num = len(gold_labels)
    # print(dataset_num)      # 50
    # category_set = Extract_category(params.segpos_alphabet.id2word, prefix_array)
    # evaluation = Eval(dataset_num)
    # evaluation.set_eval_var()
    # evaluation.eval(gold_labels, predict_labels)
    # f1_score = evaluation.calc_f1_score(params.metric)
    # category_set = Extract_category(params.segpos_alphabet.id2word)
    category_set = ('SEG', 'POS')
    dataset_num = len(gold_labels)
    # print(dataset_num)      # 50
    evaluation = Eval(category_set, dataset_num)
    evaluation.set_eval_var()
    evaluation.eval(gold_labels, predict_labels, params.metric)
    f1_score_seg, f1_score_pos = evaluation.calc_f1_score(params.metric)
    return f1_score_seg, f1_score_pos


