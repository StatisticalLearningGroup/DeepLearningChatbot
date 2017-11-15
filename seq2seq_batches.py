from __future__ import unicode_literals, print_function, division
from io import open
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

from chatbot_utils import *
from masked_cross_entropy import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
USE_CUDA = use_cuda

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

MAX_LENGTH = 10

FIRST_DATA_COL = 8

UNK_THRESH = 3

ENC_FILE = "enc.pt"
DEC_FILE = "dec.pt"
I2W_FILE = "i2w.dict"
W2I_FILE = "w2i.dict"
INF_FILE = "info.dat"
FIG_FILE = "losses.png"

DATA_DIR = "Current_Model/"

RESERVED_I2W = {SOS_INDEX: SOS, EOS_INDEX: EOS, UNK_INDEX: UNK, RMVD_INDEX: RMVD,
            PAD_INDEX: PAD}
RESERVED_W2I = dict((v,k) for k,v in RESERVED_I2W.items())

#
#   Preprocessing
#
class WordDict(object):
    def __init__(self, dicts=None):
        if dicts == None:
            self._init_dicts()
        else:
            self.word2index, self.index2word, self.word2count, self.n_words = dicts
        
    def _init_dicts(self):
        self.word2index = {}
        self.index2word = {}
        self.word2count = {}
        self.index2word.update(RESERVED_I2W)
        self.word2index.update(RESERVED_W2I)
        
        self.n_words = len(RESERVED_I2W)    #number of words in the dictionary

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if not word in RESERVED_W2I:
            if not word in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def remove_unknowns(self, cutoff):
        #find unknown words
        unks = []
        for word, count in self.word2count.items():
            if count <= cutoff:
                unks.append(word)

        #remove unknown words
        for word in unks:
            del self.index2word[self.word2index[word]]
            del self.word2index[word]
            del self.word2count[word]

        #reformat dictionaries so keys get shifted to correspond to removed words
        old_w2i = self.word2index
        self._init_dicts()
        new_index=1
        for word, index in old_w2i.items():
            self.word2index[word] = new_index
            self.index2word[new_index] = word
            new_index += 1
        self.n_words = new_index

        return unks

    def to_indices(self, words):
        indices = []
        for word in words:
            if word in self.word2index:
                indices.append(self.word2index[word])
            else:
                indices.append(self.word2index[UNK])
        return indices

    def to_words(self, indices):
        words = []
        for index in indices:
            words.append(self.index2word[index])
        return words
 
    
def readData(datafile, max_n=-1):
    print("Reading lines...")

    # Read the file and split into lines
    lines = codecs.open(datafile, encoding='utf-8', errors='ignore'). \
        read().strip().split('\n')

    if max_n > 0 and max_n < len(lines):
        lines = lines[:max_n]

    # Split every line into lines and normalize
    lines = [normalizeString(l).split(" ")[FIRST_DATA_COL:] for l in lines]

    return lines

def filterlines(lines):
    count=0
    for i in range(len(lines)):
        if len(lines[i]) > MAX_LENGTH:
            lines[i] = [RMVD]
            count += 1
    return lines, count

def prepareData(datafile, max_n=-1):
    lines = readData(datafile, max_n=max_n)
    print("Read %s sentence lines" % len(lines))
    lines, count = filterlines(lines)
    print("Trimmed %s sentences" % count)
    print("Counting words...")
    wd = WordDict()
    for line in lines:
        wd.add_sentence(line)
    print("Counted words:")
    print(wd.n_words)

    print("Removing infrequent words...")
    
    unks = wd.remove_unknowns(UNK_THRESH)
    print(str(len(unks)), "words removed.")

    return wd, lines


def getIndexPairs(wd, sentences):
    pairs = []
    msg = None
    resp = None
    addpair = False
    #print("Collecting pairs of index lists.")
    for i in range(len(sentences)):
        if i % 5000 == 0:
            pass#print(i, "pairs collected.")
        if not RMVD in sentences[i]:
            resp =  wd.to_indices(sentences[i])
            if addpair == True:
                pairs.append([msg, resp])
            msg = resp
            addpair = True
        else:
            addpair=False
    #print("Pairs of index lists collected.")
    return pairs

# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_INDEX for i in range(max_length - len(seq))]
    return seq

def random_batch(batch_size, wd, sentences):
    input_seqs = []
    target_seqs = []
    pairs = getIndexPairs(wd, sentences)

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(pair[0])
        target_seqs.append(pair[1])

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    
    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s)+1 for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)-1)+[EOS_INDEX] for s in input_seqs]
    target_lengths = [len(s)+1 for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)-1)+[EOS_INDEX] for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    
    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()
        
    return input_var, input_lengths, target_var, target_lengths



#
#   Classes for Encoder, Decoder and Attention module
#

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        
    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = torch.squeeze(hidden).dot(torch.squeeze(encoder_output))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            q = torch.cat((hidden, encoder_output))
            energy = self.attn(q.transpose())
            energy = self.v.dot(energy)
            return energy

'''class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = MAX_LENGTH
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # TODO: FIX BATCHING
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        word_embedded = self.dropout(word_embedded)
        
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        context = context.transpose(0, 1) # 1 x B x N
        
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        
        # Final output layer
        output = output.squeeze(0) # B x N
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights'''


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights



#
#   Training models
#

#helper functions for printing time elapsed and such
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

#train one epoch
def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder,
          encoder_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH, clip=50.0,
          teacher_forcing_ratio=1.0):

    #encoder_hidden = encoder.initHidden(batch_size)
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_INDEX] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()


    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t]  # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0], ec, dc

#training handler
def train_epochs(wd, lines, encoder, decoder, encoder_optimizer, decoder_optimizer,
                 batch_size, n_epochs, print_every=1000, plot_every=100, plot=True):
    ecs = []
    dcs = []
    eca = 0
    dca = 0

    epoch=0
    print_loss_total = 0
    plot_loss_total = 0

    start=time.time()

    while epoch < n_epochs:
        epoch += 1

        # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size, wd, lines)

        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer, batch_size
        )

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss
        eca += ec
        dca += dc


        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (
            time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)

        '''
        if epoch % evaluate_every == 0:
            evaluate_randomly()

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            # TODO: Running average helper
            ecs.append(eca / plot_every)
            dcs.append(dca / plot_every)
            ecs_win = 'encoder grad (%s)' % hostname
            dcs_win = 'decoder grad (%s)' % hostname
            vis.line(np.array(ecs), win=ecs_win, opts={'title': ecs_win})
            vis.line(np.array(dcs), win=dcs_win, opts={'title': dcs_win})
            eca = 0
            dca = 0
        '''


#
#   Evaluating and conversing with models
#

def evaluate(encoder, decoder, wd, input_seq, max_length=MAX_LENGTH):
    input_lengths = [len(input_seq)]
    input_seqs = [wd.to_indices(input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)

    if USE_CUDA:
        input_batches = input_batches.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_INDEX]), volatile=True)  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_INDEX:
            decoded_words.append(EOS_INDEX)
            break
        else:
            decoded_words.append(ni)

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))
        if USE_CUDA: decoder_input = decoder_input.cuda()


    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return wd.to_words(decoded_words)  # , decoder_attentions[:di + 1, :len(encoder_outputs)]

def converse(encoder, decoder, wd, max_length=MAX_LENGTH):
    print("Enter your message (press q to quit):")
    end = False
    while not end:
        msg = input()
        if "q" == msg:    #convert to curses or cmd
            end = True
        else:
            msg = normalizeString(msg).split(" ")
            raw_resp = evaluate(encoder, decoder, wd, msg)
            resp = clean_resp(raw_resp, RESERVED_W2I)
            print(resp)



#
#   Saving and loading models
#

def save_model(encoder, decoder, wd, out_path="", fig=None):
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
    pickle.dump(wd.index2word, i2w)
    i2w.close()
    w2i = open(w2i_out, 'wb')
    pickle.dump(wd.word2index, w2i)
    w2i.close()

    info = open(inf_out, 'w')
    info.write(str(encoder.hidden_size)+"\n"+str(encoder.n_layers)+"\n"+str(decoder.n_layers)+"\n"+str(wd.n_words)
               +"\n"+str(decoder.dropout))
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
    lns = info.readlines()
    hidden_size, e_layers, d_layers, n_words = [int(i) for i in lns[:4]]
    dropout = float(lns[-1])

    i2w = open(cwd+DATA_DIR+I2W_FILE, 'rb')
    w2i = open(cwd+DATA_DIR+W2I_FILE, 'rb')
    i2w_dict = pickle.load(i2w)
    w2i_dict = pickle.load(w2i)
    wd = WordDict(dicts=[w2i_dict, i2w_dict, {}, n_words])
    w2i.close()
    i2w.close()

    encoder1 = EncoderRNN(wd.n_words, hidden_size, n_layers=e_layers)
    decoder1 = LuongAttnDecoderRNN('dot', hidden_size, wd.n_words, n_layers=d_layers, dropout=dropout)
    encoder1.load_state_dict(torch.load(cwd+DATA_DIR+ENC_FILE))
    decoder1.load_state_dict(torch.load(cwd+DATA_DIR+DEC_FILE))
    encoder1.eval()
    decoder1.eval()

    tf.close()

    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()

    print("Loaded models.")

    return encoder1, decoder1, wd


#
#   Initializing and running models
#

def init_model(wd, n_layers, hidden_size, dropout=0.1, learning_rate=0.01, decoder_learning_ratio=5.0):
    encoder = EncoderRNN(wd.n_words, hidden_size, n_layers=n_layers, dropout=dropout)
    decoder= LuongAttnDecoderRNN('dot', hidden_size, wd.n_words, n_layers=n_layers, dropout=dropout)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    return encoder, decoder, encoder_optimizer, decoder_optimizer

def run_model(datafile, max_lines, n_layers, hidden_size, epochs, batch_size):
    wd, lines = prepareData(datafile, max_n=max_lines)
    encoder, decoder, enc_opt, dec_opt = init_model(wd, n_layers, hidden_size)
    train_epochs(wd, lines, encoder, decoder, enc_opt, dec_opt, batch_size, epochs)
    save_model(encoder, decoder, wd)


#
#   Command line input
#

def init_parser():
    parser = argparse.ArgumentParser(description='Sequence to sequence chatbot model.')
    parser.add_argument('-f', dest='datafile', action='store', default="movie_lines.txt")
    parser.add_argument('-m', dest='maxlines', action='store', default = 100000)
    parser.add_argument('-e', dest='epochs', action='store', default=100)
    parser.add_argument('-hs', dest='hidden_size', action='store', default=256)
    parser.add_argument('-bs', dest='batch_size', action='store', default=16)
    parser.add_argument('-l', dest='layers', action='store', default=2)
    parser.add_argument('--import', dest='model_file', action='store', default="")

    args = parser.parse_args()
    return args.datafile, args.maxlines, args.hidden_size, args.layers, args.epochs, args.batch_size, args.model_file

if __name__ == '__main__':
    datafile, max_lines, hidden_size, n_layers, epochs, batch_size, model_file = init_parser()
    max_lines, hidden_size, n_layers, epochs, batch_size = int(max_lines), int(hidden_size), int(n_layers), \
                                                           int(epochs), int(batch_size)

    if model_file == "":
        run_model(datafile, max_lines, n_layers, hidden_size, epochs, batch_size)
    else:
        encoder, decoder, wd = load_model(model_file)
        converse(encoder, decoder, wd)