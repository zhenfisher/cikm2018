#coding=utf-8
import numpy as np
import pandas as pd
import re
import os
import nltk
import pickle
from nltk.corpus import stopwords
stops1 = set(stopwords.words("spanish"))
import torch
import argparse
from datetime import datetime
from mutils import get_optimizer
from models import NLINet
from data import get_nli, get_batch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import time
from torch.utils.data.sampler import RandomSampler
from sklearn.utils import shuffle


""" Data Preprocessing """
# # Training data
# df_train_en_sp = pd.read_csv('./input/cikm_english_train_20180516.txt',sep='	', header=None,error_bad_lines=False)
# df_train_sp_en = pd.read_csv('./input/cikm_spanish_train_20180516.txt',sep='	', header=None,error_bad_lines=False)
# df_train_en_sp.columns = ['english1', 'spanish1', 'english2', 'spanish2', 'result']
# df_train_sp_en.columns = ['spanish1', 'english1', 'spanish2', 'english2','result']
# train1 = pd.DataFrame(pd.concat([df_train_en_sp['spanish1'],df_train_sp_en['spanish1']], axis=0))
# train2 = pd.DataFrame(pd.concat([df_train_en_sp['spanish2'],df_train_sp_en['spanish2']], axis=0))
# train_data = pd.concat([train1,train2],axis=1).reset_index()
# train_data = train_data.drop(['index'],axis=1)
# result = pd.DataFrame(pd.concat([df_train_en_sp['result'],df_train_sp_en['result']], axis=0)).reset_index()
# result = result.drop(['index'],axis=1)
# # pd.get_dummies(result['result']).head()
# train_data['result'] = result
#
# # Evaluation data
# test_data = pd.read_csv('./input/cikm_test_a_20180516.txt', sep='	', header=None,error_bad_lines=False)
# test_data.columns = ['spanish1', 'spanish2']
#
# def clean_sent(sent):
#     sent = sent.lower()
#     sent = re.sub(u'[_"\-;%()|+&=*%.,!?:#$@\[\]/]','',sent)
#     sent = re.sub('¡','',sent)
#     sent = re.sub('¿','',sent)
#     sent = re.sub('Á','á',sent)
#     sent = re.sub('Ó','ó',sent)
#     sent = re.sub('Ú','ú',sent)
#     sent = re.sub('É','é',sent)
#     sent = re.sub('Í','í',sent)
#     return sent
# def cleanSpanish(df):
#     df['spanish1'] = df.spanish1.map(lambda x: ' '.join([ word for word in
#                                                          nltk.word_tokenize(clean_sent(x).decode('utf-8'))]).encode('utf-8'))
#     df['spanish2'] = df.spanish2.map(lambda x: ' '.join([ word for word in
#                                                          nltk.word_tokenize(clean_sent(x).decode('utf-8'))]).encode('utf-8'))
#
# def removeSpanishStopWords(df, stop):
# 	df['spanish1'] = df.spanish1.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))
#                                                          if word not in stop]).encode('utf-8'))
# 	df['spanish2'] = df.spanish2.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))
#                                                          if word not in stop]).encode('utf-8'))
#
#
# cleanSpanish(train_data)
# removeSpanishStopWords(train_data, stops1)
# cleanSpanish(test_data)
# removeSpanishStopWords(test_data,stops1)
#
# train_data.replace('', np.nan, inplace=True)
# dirty_data = train_data[train_data.isnull().any(axis=1)]
# print dirty_data.shape[0]
# print 'positive dirty training sample:',len(dirty_data[dirty_data['result']==1])
# print 'negative dirty training sample:',len(dirty_data[dirty_data['result']==0])
#
# train_data = train_data.dropna()
# test_data.replace('', np.nan, inplace=True)
# test_data = test_data.dropna()
# print train_data.shape[0], test_data.shape[0]
#
# train_data.to_csv("input/cleaned_train.csv", index=False)
# test_data.to_csv("input/cleaned_test.csv",index = False)


""" Second Loading """

parser = argparse.ArgumentParser(description='Siamese training')

parser.add_argument("--task", type=str, default='train')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--embed_size", type=int, default=300)

params, _ = parser.parse_known_args()
print params

train_data = pd.read_csv('input/cleaned_train.csv')
test_data = pd.read_csv('input/cleaned_test.csv')
train_data.columns = ['s1','s2','label']
test_data.columns = ['s1','s2']

# 按比例分割 train dev
total_len = train_data.shape[0]
residual = total_len % params.batch_size
batch_len = (total_len - residual)/params.batch_size
train_len = int(batch_len * 0.8) * params.batch_size
valid_len = train_len + int(batch_len * 0.2) * params.batch_size
# test_len = valid_len + int (batch_len * 0.15) * params.batch_size

train = train_data.loc[0:train_len-1]
valid = train_data.loc[train_len:valid_len-1]

from collections import Counter


class Vocab(object):
    def __init__(self, all_sents, max_size=None, sos_token=None, eos_token=None, unk_token=None):
        """Initialize the vocabulary.
        Args:
            iter: An iterable which produces sequences of tokens used to update
                the vocabulary.
            max_size: (Optional) Maximum number of tokens in the vocabulary.
            sos_token: (Optional) Token denoting the start of a sequence.
            eos_token: (Optional) Token denoting the end of a sequence.
            unk_token: (Optional) Token denoting an unknown element in a
                sequence.
        """
        self.max_size = max_size
        self.pad_token = '<pad>'
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        # Add special tokens.
        id2word = [self.pad_token]
        if sos_token is not None:
            id2word.append(self.sos_token)
        if eos_token is not None:
            id2word.append(self.eos_token)
        if unk_token is not None:
            id2word.append(self.unk_token)

        # Update counter with token counts.
        counter = Counter()
        for x in all_sents:
            counter.update(x.split())

        # Extract lookup tables.
        if max_size is not None:
            counts = counter.most_common(max_size)
        else:
            counts = counter.items()
            counts = sorted(counts, key=lambda x: x[1], reverse=True)
        words = [x[0] for x in counts]
        id2word.extend(words)
        word2id = {x: i for i, x in enumerate(id2word)}

        self._id2word = id2word
        self._word2id = word2id

    def __len__(self):
        return len(self._id2word)

    def word2id(self, word):
        """Map a word in the vocabulary to its unique integer id.
        Args:
            word: Word to lookup.
        Returns:
            id: The integer id of the word being looked up.
        """
        if word in self._word2id:
            return self._word2id[word]
        elif self.unk_token is not None:
            return self._word2id[self.unk_token]
        else:
            raise KeyError('Word "%s" not in vocabulary.' % word)

    def id2word(self, id):
        """Map an integer id to its corresponding word in the vocabulary.
        Args:
            id: Integer id of the word being looked up.
        Returns:
            word: The corresponding word.
        """
        return self._id2word[id]

class SpanishDS(Dataset):

    def __init__(self, df, all_sents):
        # Assign vocabularies.
        self.s1 = df['s1'].tolist()
        self.s2 = df['s2'].tolist()
        self.label = df['label'].tolist()
        self.vocab = Vocab(all_sents, sos_token='<sos>', eos_token='<eos>', unk_token='<unk>')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Split sentence into words.
        s1_words = self.s1[idx].split()
        s2_words = self.s2[idx].split()

        # Add <SOS> and <EOS> tokens.
        s1_words = [self.vocab.sos_token] + s1_words + [self.vocab.eos_token]
        s2_words = [self.vocab.sos_token] + s2_words + [self.vocab.eos_token]

        # Lookup word ids in vocabularies.
        s1_ids = [self.vocab.word2id(word) for word in s1_words]
        s2_ids = [self.vocab.word2id(word) for word in s2_words]

        # Convert to tensors.
        s1_tensor = Variable(torch.LongTensor(s1_ids))
        s2_tensor = Variable(torch.LongTensor(s2_ids))
        label = Variable(torch.LongTensor([self.label[idx]]))

        if torch.cuda.is_available():
            s1_tensor = s1_tensor.cuda()
            s2_tensor = s2_tensor.cuda()
            label = label.cuda()

        return s1_tensor, s2_tensor, label


def get_embedding(word_dict, embedding_path, embedding_dim=300):
    # find existing word embeddings
    word_vec = {}
    with open(embedding_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}/{1} words with embedding vectors'.format(
        len(word_vec), len(word_dict)))
    missing_word_num = len(word_dict) - len(word_vec)
    missing_ratio = round(float(missing_word_num) / len(word_dict), 4) * 100
    print('Missing Ratio: {}%'.format(missing_ratio))

    # handling unknown embeddings
    for word in word_dict:
        if word not in word_vec:
            # If word not in word_vec, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            word_vec[word] = new_embedding
    print "Filled missing words' embeddings."
    print "Embedding Matrix Size: ", len(word_vec)

    return word_vec

def save_embed(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print 'Embedding saved'
def load_embed(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


train = shuffle(train)
valid = shuffle(valid)
all_sents = train['s1'].tolist() + train['s2'].tolist() + valid['s1'].tolist() + valid['s2'].tolist()

trainDS = SpanishDS(train, all_sents)
validDS = SpanishDS(valid, all_sents)


full_embed_path = 'input/wiki.es.vec'
cur_embed_path = 'input/embedding.pkl'
if os.path.exists(cur_embed_path):
    embed_dict = load_embed(cur_embed_path)
    print 'Loaded existing embedding.'
else:
    embed_dict = get_embedding(trainDS.vocab._id2word, full_embed_path)
    save_embed(embed_dict,cur_embed_path)
    print 'Saved generated embedding.'

embed_dim = 300
vocab_size = len(embed_dict)
# initialize nn embedding
embedding = nn.Embedding(vocab_size, embed_dim)
embed_list = []
for word in trainDS.vocab._id2word:
    embed_list.append(embed_dict[word])
weight_matrix = np.array(embed_list)
# pass weights to nn embedding
embedding.weight = nn.Parameter(torch.from_numpy(weight_matrix).type(torch.FloatTensor), requires_grad = False)


""" Model """
# model config
config = {
    'vocab_size'     :  len(embed_dict)  ,
    'hidden_size'    :  150            ,
    'batch_size'     :  1             ,
    'embed_size'     :  300            ,
    'num_layers'     :  1              ,
    'bidirectional'  :  False ,
    'embedding_matrix': embedding,
}


class EncoderRNN(nn.Module):
    def __init__(self, config):
        super(EncoderRNN, self).__init__()
        self.hidden_size = config['hidden_size']
        self.batch_size = config['batch_size']
        self.embed_size = config['embed_size']
        self.num_layers = config['num_layers']
        self.bidir = config['bidirectional']

        self.embedding = config['embedding_matrix']
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidir)

    def initHiddenCell(self):
        zero_hidden = Variable(torch.randn(1, self.batch_size, self.hidden_size))
        zero_cell = Variable(torch.randn(1, self.batch_size, self.hidden_size))
        return zero_hidden, zero_cell

    def forward(self, input, hidden, cell):
        input = self.embedding(input).view(1, 1, -1)
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        return output, hidden, cell

class DecoderRNN(nn.Module):
    def __init__(self, config):
        super(DecoderRNN, self).__init__()
        self.hidden_size = config['hidden_size']
        self.batch_size = config['batch_size']
        self.embed_size = config['embed_size']
        self.num_layers = config['num_layers']
        self.vocab_size = config['vocab_size']
        self.bidir = config['bidirectional']

        self.embedding = config['embedding_matrix']
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidir)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

    def initHiddenCell(self):
        zero_hidden = Variable(torch.randn(1, self.batch_size, self.hidden_size))
        zero_cell = Variable(torch.randn(1, self.batch_size, self.hidden_size))
        return zero_hidden, zero_cell

    def forward(self, input, hidden, cell):
        input = self.embedding(input).view(1, 1, -1)
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output = self.fc(output)
        output = F.log_softmax(output, dim=-1)
        return output, hidden, cell

encoder = EncoderRNN(config)
decoder = DecoderRNN(config)

loss_weights = torch.ones(config['vocab_size'])
loss_weights[0] = 0
if torch.cuda.is_available():
    loss_weights = loss_weights.cuda()
criterion = torch.nn.NLLLoss(loss_weights)


learning_rate = 0.05
# encoder_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, encoder.parameters()) ,
#                                     lr=learning_rate)
# decoder_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, decoder.parameters()),
#                                     lr=learning_rate)
encoder_optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, encoder.parameters()) ,
                                    lr=learning_rate)
decoder_optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, decoder.parameters()),
                                    lr=learning_rate)

train_log_string = '%s :: Epoch %i :: Iter %i / %i :: train loss: %0.4f'
valid_log_string = '\n%s :: Epoch %i :: valid loss: %0.4f'
start_time = time.time()



ckpt_dir = 'input/'
# Restore saved model (if one exists).
ckpt_path = os.path.join(ckpt_dir, 'model.pt')
if os.path.exists(ckpt_path):
    print('Loading checkpoint: %s' % ckpt_path)
    ckpt = torch.load(ckpt_path)
    epoch = ckpt['epoch']
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    encoder_optimizer.load_state_dict(ckpt['encoder_optimizer'])
    decoder_optimizer.load_state_dict(ckpt['decoder_optimizer'])
else:
    epoch = 0

""" Training """
print 'Start training ...'
while epoch < 5:
    # Train
    train_loss = []
    train_sampler = RandomSampler(trainDS)
    for idx, train_idx in enumerate(train_sampler):
        # print('train_idx', train_idx, 'i' , i)
        s1, s2, label = trainDS[train_idx]
        s1_length = s1.size()[0]
        s2_length = s2.size()[0]
    #     print train_idx
    #     print s1
    #     print s2
    #     print label

        # Clear gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # feed s1 into the encoder, one word a time
        h, c = encoder.initHiddenCell()

        # 可以倒序输入，有论文说效果可能变好
    #     for i in reversed(range(s1_length)):
        for i in range(s1_length):
    #         print 'Enc Input:', s1[i]
            input = torch.LongTensor([s1[i]])
    #         print embedding(input).shape
            input, h, c = encoder(input, h, c)
    #     print 'hidden context size:',input.shape

        # decode
        loss = 0
        for j in range(s2_length-1):
    #         print 'Dec Input:', s2[j]
            input = torch.LongTensor([s2[j]])
            output, h, c = decoder(input, h, c)
            output = output.squeeze(0)
            loss += criterion(output, Variable(torch.LongTensor([s2[j+1].tolist()])))
    #         print loss

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        train_loss.append(loss.data.cpu())

        # Every once and a while check on the loss
        if ((idx+1) % 1000) == 0:
            print(train_log_string % (datetime.now(), epoch, idx+1, len(train), np.mean(train_loss)))
            train_loss = []

    # Valid
    print 'Validating...'
    valid_loss = []
    valid_sampler = RandomSampler(validDS)
    for i, valid_idx in enumerate(valid_sampler):
        # print('train_idx', train_idx, 'i' , i)
        s1,s2,label = validDS[valid_idx]
        s1_length = s1.size()[0]
        s2_length = s2.size()[0]

        # encode
        # 可以倒序输入，有论文说效果可能变好
    #     for i in reversed(range(s1_length)):
        for i in range(s1_length):
            input = torch.LongTensor([s1[i]])
            input, h, c = encoder(input, h, c)

        # decode
        loss = 0
        for j in range(s2_length-1):
            input = torch.LongTensor([s2[j]])
            output, h, c = decoder(input, h, c)
            output = output.squeeze(0)
            loss += criterion(output, Variable(torch.LongTensor([s2[j+1].tolist()])))
    #         print loss

        valid_loss.append(loss.data.cpu())

    print(valid_log_string % (datetime.now(), epoch, np.mean(valid_loss)))

    state_dict = {
        'epoch': epoch,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict()
    }
    torch.save(state_dict, ckpt_path)
    print 'Model saved'

    epoch += 1

elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print("Time consumed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))