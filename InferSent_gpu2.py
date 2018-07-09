#coding=utf-8
import numpy as np
import pandas as pd
import os
import re
import nltk
import argparse

import time

import torch
from torch.autograd import Variable
import torch.nn as nn
from models import NLINet
from mutils import get_optimizer
from data import get_nli, get_batch

from nltk.stem import SnowballStemmer
snowball_stemmer1 = SnowballStemmer('spanish')
snowball_stemmer2 = SnowballStemmer('english')
snowball_stemmer1.stem
snowball_stemmer2.stem

# from nltk.corpus import stopwords
# stops1 = set(stopwords.words("spanish"))
# stops2 = set(stopwords.words("english"))



#################### READ DATA ####################

df_train_en_sp = pd.read_csv('./input/cikm_english_train_20180516.txt',sep='	', header=None,error_bad_lines=False)
df_train_sp_en = pd.read_csv('./input/cikm_spanish_train_20180516.txt',sep='	', header=None,error_bad_lines=False)
df_train_en_sp.columns = ['english1', 'spanish1', 'english2', 'spanish2', 'result']
df_train_sp_en.columns = ['spanish1', 'english1', 'english2', 'spanish2', 'result']

train1 = pd.DataFrame(pd.concat([df_train_en_sp['spanish1'],df_train_sp_en['spanish1']], axis=0))
train2 = pd.DataFrame(pd.concat([df_train_en_sp['spanish2'],df_train_sp_en['spanish2']], axis=0))
train = pd.concat([train1,train2],axis=1).reset_index()
train = train.drop(['index'],axis=1)
result = pd.DataFrame(pd.concat([df_train_en_sp['result'],df_train_sp_en['result']], axis=0)).reset_index()
result = result.drop(['index'],axis=1)
# pd.get_dummies(result['result']).head()
train['result'] = result



# Evaluation data
infer_data = pd.read_csv('./input/cikm_test_a_20180516.txt', sep='	', header=None,error_bad_lines=False)
infer_data.columns = ['spanish1', 'spanish2']


print("Read Data Done!")
# train.head()


#################### CLEAN TEXT ####################

# def subchar(text):
# 	text=text.replace("á", "a")
# 	text=text.replace("ó", "o")
# 	text=text.replace("é", "e")
# 	text=text.replace("í", "i")
# 	text=text.replace("ú", "u")
# 	return text
#西班牙语缩写还原#
def stemSpanish(df):
	df['spanish1'] = df.spanish1.map(lambda x: ' '.join([snowball_stemmer1.stem(word) for word in
                                                         nltk.word_tokenize(x.lower().decode('utf-8'))]).encode('utf-8'))
	df['spanish2'] = df.spanish2.map(lambda x: ' '.join([snowball_stemmer1.stem(word) for word in
                                                         nltk.word_tokenize(x.lower().decode('utf-8'))]).encode('utf-8'))
def stemEnglish(df):
    df['english1'] =  df.english1.map(lambda x:' '.join([snowball_stemmer2.stem(word) for word in
                                                        nltk.word_tokenize(x.lower().decode('utf-8'))]).encode('utf-8'))
    df['english2'] =  df.english2.map(lambda x:' '.join([snowball_stemmer2.stem(word) for word in
                                                        nltk.word_tokenize(x.lower().decode('utf-8'))]).encode('utf-8'))
# def removeSpanishStopWords(df, stop):
#     df['spanish1'] = df.spanish1.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))
#                                                          if word not in stops1]).encode('utf-8'))
#     df['spanish2'] = df.spanish2.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))
#                                                          if word not in stops1]).encode('utf-8'))
# def removeEnglishStopWords(df, stop):
#     df['english1'] = df.english1.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))
#                                                          if word not in stops2]).encode('utf-8'))
#     df['english2'] = df.english2.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))
#                                                          if word not in stops2]).encode('utf-8'))
def removeEnglishSigns(df):
    df['english1'] = df.english1.map(lambda x: re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', '', x))
    df['english2'] = df.english2.map(lambda x: re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', '', x))
def removeSpanishSigns(df):
    df['spanish1'] = df.spanish1.map(lambda x: re.sub(r'[_"\-;%()|+&=*%.,¡!¿?:#$@\[\]/]', '', x))
    df['spanish2'] = df.spanish2.map(lambda x: re.sub(r'[_"\-;%()|+&=*%.,¡!¿?:#$@\[\]/]', '', x))

# 处理 Spanish
# train
stemSpanish(train)
removeSpanishSigns(train)

# eval
stemSpanish(infer_data)
removeSpanishSigns(infer_data)


train_data = train
train_data.columns = ['s1','s2','label']
infer_data.columns = ['s1','s2']

# 按比例分割 train dev test
total_len = train_data.shape[0]
train_len = int(total_len * 0.8)
dev_len = train_len + int(total_len * 0.15)
test_len = dev_len + int(total_len * 0.05)

train = train_data.loc[0:train_len-1]
valid = train_data.loc[train_len:dev_len-1]
test = train_data.loc[dev_len:]

print("Clean Text and Split Done!")
# W2V_PATH = 'input/wiki.es.vec'
#
# word_vec = build_vocab(train['s1'].tolist() + train['s2'].tolist() +
#                        valid['s1'].tolist() + valid['s2'].tolist() +
#                        test['s1'].tolist() + test['s2'].tolist(), W2V_PATH)

''' Vocab '''

def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    word_dict['<unk>'] = ''
    return word_dict
all_sent = train_data['s1'].tolist() + train_data['s2'].tolist() + infer_data['s1'].tolist() + infer_data['s2'].tolist()
word_dict = get_word_dict(all_sent)


''' Embedding '''

def get_embedding(word_dict, embedding_path):
    # create word_vec with embedding vectors
    word_vec = {}
    with open(embedding_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}/{1} words with embedding vectors'.format(
                len(word_vec), len(word_dict)))
    missing_word_num = len(word_dict) - len(word_vec)
    missing_ratio = round(float(missing_word_num)/len(word_dict),4)*100
    print('Missing Ratio: {}%'.format(missing_ratio))
    return word_vec

embedding_path = 'input/wiki.es.vec'
word_vec = get_embedding(word_dict, embedding_path)



''' Token '''

vocab_to_int = {}
value = 0
for word in word_dict:
    vocab_to_int[word] = value
    value += 1

int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

''' Handling unknown tokens'''
embedding_dim = 300
for word, i in vocab_to_int.items():
    if word not in word_vec:
        # If word not in word_vec, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        word_vec[word] = new_embedding
print "Filled Unk words' embeddings"


''' adding sos, eos; split and filter words'''

# train and testdata
for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split].tolist()])

# evaluation data
for split in ['s1', 's2']:
    infer_data[split] = np.array([['<s>'] +
        [word for word in sent.split() if word in word_vec] +
        ['</s>'] for sent in infer_data[split].tolist()])

''' Model '''


parser = argparse.ArgumentParser(description='InferSent training')
# paths
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model_2.pickle')


# training
parser.add_argument("--n_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='InferSent', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=2, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
params, _ = parser.parse_known_args()
params.word_emb_dim = 300

# # gpu
# parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID")
# parser.add_argument("--seed", type=int, default=1234, help="seed")
#
#
# # set gpu device
# torch.cuda.set_device(params.gpu_id)

print params



"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,

}

# model
encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
nli_net = NLINet(config_nli_model)
print(nli_net)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(nli_net.parameters(), **optim_params)


# cuda by default
if torch.cuda.is_available():
    nli_net.cuda()
    loss_fn.cuda()

""" TRAIN """

def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]


    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size].tolist(),
                                     word_vec)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size].tolist(),
                                     word_vec)
        if torch.cuda.is_available():
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size].tolist())).cuda()
        else:
            s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
            tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size].tolist()))

        k = s1_batch.size(1)  # actual batch size

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss
        loss = loss_fn(output, tgt_batch)
        #
        all_costs.append(loss.data[0])
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.*correct/(stidx+k), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    return train_acc

def evaluate(epoch,eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size].tolist(), word_vec)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size].tolist(), word_vec)
        if torch.cuda.is_available():
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size].tolist())).cuda()
        else:
            s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
            tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size].tolist()))

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * correct / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            state_dict = {
                'epoch': epoch,
                'model': nli_net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state_dict, os.path.join(params.outputdir,
                       params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc


val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None

"""
Train model on Natural Language Inference task
"""
# Restore saved model (if one exists).
ckpt_path = os.path.join(params.outputdir, params.outputmodelname)
if os.path.exists(ckpt_path):
    print('Loading checkpoint: %s' % ckpt_path)
    ckpt = torch.load(ckpt_path)
    epoch = ckpt['epoch']
    nli_net.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
else:
    epoch = 1

while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1

# Run best model on test set.
# nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))
nli_net.load_state_dict(ckpt['model'])


print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, 'valid', True)
evaluate(0, 'test', True)


'''Inference'''

def inference(infer_data):
    nli_net.eval()
    prob_res_1 = []
    s1 = infer_data['s1']
    s2 = infer_data['s2']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size].tolist(), word_vec)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size].tolist(), word_vec)
        s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        # get softmax probability
        sm = nn.Softmax()
        res = sm(output.data)[:,1]
        prob_res_1 += res.data.tolist()
        return prob_res_1


''' Inference And Write Result'''
result = inference(infer_data)
result = pd.DataFrame(result).T
result.to_csv('result1.txt',header=False,index=False)

# Save encoder instead of full model
torch.save(nli_net.encoder.state_dict(), os.path.join(params.outputdir, params.outputmodelname + '.encoder.pkl'))
