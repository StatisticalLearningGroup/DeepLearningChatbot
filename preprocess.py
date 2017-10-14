import numpy as np
import codecs


PAD = "<PAD>"
EOS = "<EOS"
GO = "<GO>"
UNK = "<UNK>"

RESERVED = {PAD: 0, EOS: 1, GO: 2, UNK: 3}

FREQ_CUTOFF = 3

RAW_DATA_FILE = "movie_lines.txt"
FIRST_DATA_COL = 8

#BUCKETS = [(5, 10), (10, 15), (20, 25), (40, 50)]



def one_hot_encode(i, n):
    return [int(j==i) for j in range(n)]

def get_stripped_data(filename):
    data = []
    with codecs.open(filename, "r", encoding='utf-8', errors='ignore') as infile:
        for line in infile:
            data.append(line.split(" ")[FIRST_DATA_COL:])
    return data

def prepare_data(stripped_data):
    word_dict = {}
    freq = {}
    maxlen = 0
    data = []


    #fill data list and dictionary
    for line in stripped_data:
        msg = []
        for word in line:
            if word in word_dict:
                freq[word] += 1
            else:
                word_dict[word] = len(word_dict) + 3
                freq[word] = 1
            msg.append(word_dict[word])
        data.append(msg)
        maxlen = max(maxlen, len(msg))

    #find unknown words
    unk_words = []
    for word in word_dict:
        if freq[word] < FREQ_CUTOFF:
            unk_words.append(word_dict[word])

    word_dict.update(RESERVED)

    #add reserved tokens to data (e.g. EOS, GO, PAD, UNK)
    for j in range(len(data)):
        for i in range(len(data[j])):
            if data[j][i] in unk_words:
                data[j][i] = word_dict[UNK]
        '''
        c=0
        while len(data[j]) > BUCKETS[c][0]:
            if c >= len(BUCKETS):
                raise Exception("Buckets too small.")
            c += 1
        b = max(BUCKETS[c][0] - len(data[j]), 0)
        '''
        b = max(maxlen - len(data[j]), 0)
        data[j] = [word_dict[GO]] + data[j] + [word_dict[PAD]]*b + [word_dict[EOS]]

    #remove unknown words from dict:
    for word in unk_words:
        del word_dict[word]

    return data, word_dict, maxlen

def get_mr_pairs(data, word_dict, maxlen):
    n_pairs = len(data)-1
    dim = len(word_dict)
    MR_pairs = []#np.zeros([n_pairs, 2, maxlen, dim])

    msg = None

    for line in data:
        rep = []
        for word in line:
            rep.append(one_hot_encode(word, dim))
        if msg != None:
            MR_pairs.append([msg, rep])
        msg = rep

    return np.array(MR_pairs)


DATA_CUTOFF = 1000

stripped_data = get_stripped_data(RAW_DATA_FILE)[:DATA_CUTOFF]
data, word_dict, maxlen = prepare_data(stripped_data)
