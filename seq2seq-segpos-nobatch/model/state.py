# class state:
#     def __init__(self, gold, chars):
#         self.chars = chars
#         self.gold = gold
#
#         self.action = []
#         self.pos_record = []
#         self.pos_index = []
#         self.words_record = []


class state:
    def __init__(self, gold, chars):
        self.m_chars = chars
        self.m_gold = gold
        self.words = []
        self.pos_id = []
        self.pos_labels = []
        self.actions = []

        self.word_hiddens = []
        self.word_cells = []



class state_nowordlstm:
    def __init__(self, gold, chars):
        self.chars = chars
        self.gold = gold

        self.words = []
        self.pos_id = []
        self.pos_labels = []
        self.actions = []

        self.word_hiddens = []
        self.word_cells = []

        self.all_h = []
        self.all_c = []


class state_batch:
    def __init__(self, golds, chars, batch_size, max_length):
        temp = [[] for _ in range(max_length)]
        for i in range(batch_size):
            for j in range(max_length):
                temp[j].append(chars[i][j])
        self.chars = temp

        temp = [[] for _ in range(max_length)]
        for i in range(batch_size):
            for j in range(max_length):
                temp[j].append(golds[i][j])
        self.golds = temp

        self.action = [[] for _ in range(batch_size)]
        # self.pos_record = [[] for _ in range(max_length)]
        # self.pos_index = [[] for _ in range(max_length)]
        # self.words_record = [[] for _ in range(max_length)]
        # self.pos_record = []
        # self.pos_index = []
        # self.words_record = []
        self.pos_record = [[] for _ in range(batch_size)]
        self.pos_index = [[] for _ in range(batch_size)]
        self.words_record = [[] for _ in range(batch_size)]


