

from __future__ import unicode_literals, print_function, division

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from io import open
import unicodedata
import re
import random
import codecs
import argparse
import pickle
import tarfile
import datetime
import os
import time
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


SOS_INDEX = 0
EOS_INDEX = 1
UNK_INDEX = 2
RMVD_INDEX = 3
PAD_INDEX = 4

UNK = "UNK"
RMVD = "RMVD"
EOS = "EOS"
SOS = "SOS"
PAD = "PAD"

RESERVED = {SOS_INDEX: SOS, EOS_INDEX: EOS, UNK_INDEX: UNK, RMVD_INDEX: RMVD, PAD_INDEX: PAD}

MAX_LENGTH = 10
MAX_RESPONSE_LENGTH = 20

FIRST_DATA_COL = 8

UNK_THRESH = 3

ENC_FILE = "enc.pt"
DEC_FILE = "dec.pt"
I2W_FILE = "i2w.dict"
W2I_FILE = "w2i.dict"
INF_FILE = "info.dat"
FIG_FILE = "losses.png"

DATA_DIR = "Current_Model/"

#dictionary class - sorry, I know I'm misusing the word Corpus but I only realized that now and dont want to refactor it

class Corpus:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.word2index.update(RESERVED)
        self.index2word.update(RESERVED)
        self.n_words = len(RESERVED)

    def insert_data(self, w2i, i2w, n_w):
        self.word2index = w2i
        self.index2word = i2w
        self.n_words = n_w

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


#read the data, store the lines in a list, remove overly long messages and strip out unnecessary characters (like punctuation)

def readData(datafile, max_n=-1):
    print("Reading lines...")

    # Read the file and split into lines
    lines = codecs.open(datafile, encoding='utf-8', errors='ignore'). \
        read().strip().split('\n')

    if max_n > 0 and max_n < len(lines):
        lines = lines[:max_n]

    # Split every line into lines and normalize
    lines = [normalizeString(l).split(" ")[FIRST_DATA_COL:] for l in lines]

    corpus = Corpus()

    return corpus, lines






def filterlines(lines):
    for i in range(len(lines)):
        if len(lines[i]) > MAX_LENGTH:
            lines[i] = [RMVD]
    return lines



def prepareData(datafile, max_n=-1):
    corpus, lines = readData(datafile, max_n=max_n)
    print("Read %s sentence lines" % len(lines))
    lines = filterlines(lines)
    print("Trimmed to %s sentences" % len(lines))
    print("Counting words...")
    for line in lines:
        corpus.addSentence(line)
    print("Counted words:")
    print(corpus.n_words)

    print("Removing infrequent words...")
    unks = []
    rm_count=0
    for word, count in corpus.word2count.items():
        if count < UNK_THRESH:
            unks.append(word)
    print(len(unks), "words removed from dictionary.")
    for i in range(len(lines)):
        if i % 5000 == 0:
            print("Line", i, "parsed.")
        for j in range(len(lines[i])):
            if lines[i][j] in unks:
                lines[i][j] = UNK
                rm_count += 1
    print(rm_count, "words removed from corpus.")

    return corpus, lines




class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



def indexesFromSentence(corpus, sentence):
    indices = []
    for word in sentence:
        index = -1
        try:
            index = corpus.word2index[word]
        except KeyError:
            index = UNK_INDEX
        finally:
            indices.append(index)
    return indices


def variableFromSentence(corpus, sentence):
    indexes = indexesFromSentence(corpus, sentence)
    indexes.append(EOS_INDEX)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def getTrainingPairs(lang, sentences):
    pairs = []
    i_rmvd = -1
    msg = None
    resp = None
    addpair = False
    print("Collecting training pairs.")
    for i in range(len(sentences)):
        if i % 5000 == 0:
            print(i, "pairs collected.")
        if not RMVD in sentences[i]:
            rep = variableFromSentence(lang, sentences[i])
            if addpair == True:
                pairs.append([msg, rep])
            msg = rep
            addpair = True
        else:
            addpair=False
    print("Training pairs collected.")
    return pairs




teacher_forcing_ratio = 0.5


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_RESPONSE_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    #encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    #encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[SOS_INDEX]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_INDEX:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#




def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def trainIters(corpus, lines, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, plot=True):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = getTrainingPairs(corpus, lines)
    n = len(training_pairs)
    criterion = nn.NLLLoss()

    print("Beginning training.")
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[(iter - 1) % n]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    if plot:
        return showPlot(plot_losses)



def showPlot(points):
    plt.figure(frameon=False)
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    return plt.gcf()


def evaluate(encoder, decoder, corpus, sentence, max_length=MAX_RESPONSE_LENGTH):
    input_variable = variableFromSentence(corpus, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[SOS_INDEX]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_INDEX:
            decoded_words.append(EOS)
            break
        else:
            decoded_words.append(corpus.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words

def converse(encoder, decoder, corpus, max_length = MAX_LENGTH):
    print("Enter your message:")
    end = False
    while not end:
        msg = input()
        if "exit" in msg:
            end=True
        else:
            msg = normalizeString(msg).split(" ")
            resp = evaluate(encoder, decoder, corpus, msg)
            print(resp)

#read command line input and arguments
def init_parser():
    parser = argparse.ArgumentParser(description='Sequence to sequence chatbot model.')
    parser.add_argument('-f', dest='datafile', action='store', default="movie_lines.txt")
    parser.add_argument('-m', dest='maxlines', action='store', default = 100000)
    parser.add_argument('-i', dest='iters', action='store', default=100000)
    parser.add_argument('-hs', dest='hidden_size', action='store', default=256)
    parser.add_argument('--import', dest='model_file', action='store', default="")

    args = parser.parse_args()
    return args.datafile, args.maxlines, args.iters, args.hidden_size, args.model_file

#run the model with given parameters
def run_model(datafile, hidden_size = 256, iters=100000, max_n = 100000):
    corpus, lines = prepareData(datafile, max_n=max_n)

    encoder1 = EncoderRNN(corpus.n_words, hidden_size)
    decoder1 = DecoderRNN(hidden_size, corpus.n_words)
    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()
    plot = trainIters(corpus, lines, encoder1, decoder1, iters, print_every=5000)

    save_model(encoder1, decoder1, corpus, fig=plot)


#save/load the model to/from a tar file
def save_model(encoder, decoder, corpus, out_path="", fig=None):
    print("Saving models.")

    cwd = os.getcwd() + '/'

    enc_out = out_path+ENC_FILE
    dec_out = out_path+DEC_FILE
    i2w_out = out_path+I2W_FILE
    w2i_out = out_path+W2I_FILE
    inf_out = out_path+INF_FILE
    fig_out = out_path+FIG_FILE

    torch.save(encoder.state_dict(), enc_out)
    torch.save(decoder.state_dict(), dec_out)

    i2w = open(i2w_out, 'wb')
    pickle.dump(corpus.index2word, i2w)
    i2w.close()
    w2i = open(w2i_out, 'wb')
    pickle.dump(corpus.word2index, w2i)
    w2i.close()

    info = open(inf_out, 'w')
    info.write(str(encoder.hidden_size)+"\n"+str(encoder.n_layers)+"\n"+str(decoder.n_layers)+"\n"+str(corpus.n_words))
    info.close()

    if fig != None:
        fig.savefig(fig_out)

    print("Bundling models")
    t = datetime.datetime.now()
    timestamp = str(t.day) + "_" + str(t.hour) + "_" + str(t.minute)
    tf = tarfile.open(cwd+out_path +"s2s_" + timestamp + ".tar", mode='w')
    tf.add(enc_out)
    tf.add(dec_out)
    tf.add(i2w_out)
    tf.add(w2i_out)
    tf.add(inf_out)
    if fig != None:
        tf.add(fig_out)
    tf.close()

    os.remove(enc_out)
    os.remove(dec_out)
    os.remove(i2w_out)
    os.remove(w2i_out)
    os.remove(inf_out)
    if fig != None:
        os.remove(fig_out)

    print("Finished saving models.")

def load_model(model_file):
    print("Loading models.")
    cwd = os.getcwd()+'/'
    tf = tarfile.open(model_file)
    tf.extractall(path=DATA_DIR)
    info = open(cwd+DATA_DIR+INF_FILE, 'r')
    hidden_size, e_layers, d_layers, n_words = [int(i) for i in info.readlines()]

    i2w = open(cwd+DATA_DIR+I2W_FILE, 'rb')
    w2i = open(cwd+DATA_DIR+W2I_FILE, 'rb')
    corpus = Corpus()
    i2w_dict = pickle.load(i2w)
    w2i_dict = pickle.load(w2i)
    corpus.insert_data(w2i_dict, i2w_dict, n_words)
    w2i.close()
    i2w.close()

    encoder1 = EncoderRNN(corpus.n_words, hidden_size)
    decoder1 = DecoderRNN(hidden_size, corpus.n_words)
    encoder1.load_state_dict(torch.load(cwd+DATA_DIR+ENC_FILE))
    decoder1.load_state_dict(torch.load(cwd+DATA_DIR+DEC_FILE))
    encoder1.eval()
    decoder1.eval()

    tf.close()

    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()

    print("Loaded models.")

    return encoder1, decoder1, corpus

encoder1 = None
decoder1 = None
corpus = None

if __name__ == '__main__':
    datafile, maxlines, iters, hidden_size, model_file = init_parser()

    if model_file == "":
        run_model(datafile, hidden_size=int(hidden_size), iters=int(iters), max_n=int(maxlines))
    else:
        encoder1, decoder1, corpus = load_model(model_file)
        converse(encoder1, decoder1, corpus)




