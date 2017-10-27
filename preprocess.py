import numpy as np
import codecs


PAD = "<PAD>"
EOS = "<EOS"
GO = "<GO>"
UNK = "<UNK>"
RMVD = "<RMVD>"

RESERVED = {PAD: 0, EOS: 1, GO: 2, UNK: 3, RMVD: 4}

FREQ_CUTOFF = 3
LENGTH_CUTOFF = 20

RAW_DATA_FILE = "movie_lines.txt"
FIRST_DATA_COL = 8

#BUCKETS = [(5, 10), (10, 15), (20, 25), (40, 50)]



def one_hot_encode(i, n):
    return [int(j==i) for j in range(n)]

def get_stripped_data(filename):
    data = []
    with codecs.open(filename, "r", encoding='utf-8', errors='ignore') as infile:
        for line in infile:
            linedat = line.split(" ")[FIRST_DATA_COL:]
            if len(linedat <= LENGTH_CUTOFF):
                data.append(linedat)
            else:
                data.append([RMVD])

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
                word_dict[word] = len(word_dict) + (len(RESERVED) - 1)
                freq[word] = 1
            msg.append(word_dict[word])
        data.append(msg)
        maxlen = max(maxlen, len(msg))

    #find unknown words
    unk_words = []
    for word in word_dict:
        if freq[word] < FREQ_CUTOFF:
            unk_words.append([word, word_dict[word]])

    word_dict.update(RESERVED)

    #add reserved tokens to data (e.g. EOS, GO, PAD, UNK)
    for j in range(len(data)):
        for i in range(len(data[j])):
            for word in unk_words:
                if data[j][i] == word[1]:
                    data[j][i] = word_dict[UNK]


        b = max(maxlen - len(data[j]), 0)
        data[j] = [word_dict[GO]] + data[j] + [word_dict[PAD]]*b + [word_dict[EOS]]

    #remove unknown words from dict:
    for word in unk_words:
        del word_dict[word[0]]

    return data, word_dict, maxlen

def get_mr_pairs(data, word_dict, maxlen):
    n_pairs = len(data)-1
    dim = len(word_dict)
    MR_pairs = []#np.zeros([n_pairs, 2, maxlen, dim])

    msg = None

    for line in data:
        if not line.contains(RMVD):
            rep = []
            for word in line:
                rep.append(one_hot_encode(word, dim))
            if msg != None:
                MR_pairs.append([msg, rep])
            msg = rep
        else:
            msg = None

    return np.array(MR_pairs)


DATA_CUTOFF = 10000

stripped_data = get_stripped_data(RAW_DATA_FILE)[:DATA_CUTOFF]
data, word_dict, maxlen = prepare_data(stripped_data)
