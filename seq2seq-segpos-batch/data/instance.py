from collections import Counter
import string

class Instance:
    def __init__(self):
        self.words = []
        self.chars = []
        self.extchars = []

        self.left_bichars = []
        self.right_bichars = []
        self.extleft_bichars = []
        self.extright_bichars = []

        self.char_type = []
        self.word_seg = []          # format:[2,3]
        self.word_seg_pos = []      # format:[2,3]NN

        self.gold_action = []              # format:'SEP#NN'
        self.gold_pos = []               # format:'NN'
        self.word_len = []

        self.last_wordlen = []
        self.last_pos = []

        self.last_wordlen_index = []
        self.last_pos_index = []

        self.words_index = []
        self.chars_index = []
        self.extchars_index = []

        self.left_bichar_index = []
        self.right_bichar_index = []
        self.extleft_bichar_index = []
        self.extright_bichar_index = []

        self.pos_index = []
        self.char_type_index = []
        self.action_index = []


def char_type(char):
    # char = char.unicode('utf-8')
    char_len = len(char.encode('utf-8'))
    char_type = ''
    if char_len > 2:
        char_type = 'U'
    elif char_len == 2:
        char_type = 'u'
    elif char.isalpha():        #####
        if char.isupper():
            char_type = 'E'
        else:
            char_type = 'e'
    elif char.isdigit():
        char_type = 'd'
    elif char in string.punctuation:       #####
        char_type = 'p'
    else:
        char_type = 'o'
    return char_type


if __name__=='__main__':
    char_type('我')
