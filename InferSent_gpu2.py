#coding=utf-8
import numpy as np
import pandas as pd
import os
import re
import nltk
import argparse

import time

import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
from models import NLINet
from mutils import get_optimizer
from data import get_nli, get_batch


parser = argparse.ArgumentParser(description='InferSent training')
# paths
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model_5.pickle')


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


print params

##################### LOAD EMBEDDINGS ########################

def load_embed(name ):
    with open('input/' + name , 'rb') as f:
        return pickle.load(f)

word_vec = load_embed('embedding.pkl')

print 'Embedding - Loaded!'



##################### LOAD DATA ########################

train_data = pd.read_csv('input/cleaned_train.csv')
infer_data = pd.read_csv('input/cleaned_test.csv')

train_data = train_data.dropna()
infer_data = infer_data.dropna()
print 'After cleaning up null: ', 'Train data size:', train_data.shape[0], 'Inference data size:', infer_data.shape[0]

# 按比例分割 train dev test
total_len = train_data.shape[0]
residual=total_len%params.batch_size
batch_len = (total_len - residual)/params.batch_size
train_len = int ( batch_len* 0.8) * params.batch_size
valid_len = train_len + int (batch_len * 0.15) * params.batch_size
test_len = valid_len + int (batch_len * 0.15) * params.batch_size

train = train_data.loc[0:train_len-1]
valid = train_data.loc[train_len:valid_len-1]
test = train_data.loc[valid_len:]

print len(train)
print len(valid)
print len(test)


# adding sos, eos; split and filter words

# inference data
for split in ['s1', 's2']:
    infer_data[split] = np.array([['<s>'] +
        [word for word in sent.split() if word in word_vec] +
        ['</s>'] for sent in infer_data[split].tolist()])

# train data
for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split].tolist()])
        # for sent in eval(data_type)[split].tolist():
        #     body = []
        #     for word in sent.split():
        #         if word in word_vec:
        #             body.append(word)
        #     res = np.array([['<s>'] + body +['</s>']])
        # totol_res.append(res)




print 'Data - Loaded!'



##################### MODEL ########################


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


encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
nli_net = NLINet(config_nli_model)
print(nli_net)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.NLLLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(nli_net.parameters(), **optim_params)


# cuda by default
if torch.cuda.is_available():
    nli_net.cuda()
    loss_fn.cuda()

##################### TRAINING ########################

def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data

    # permutation = np.random.permutation(len(train['s1']))
    #
    # target = train['label'][permutation]
    # s1 = train['s1'][permutation]
    # s2 = train['s2'][permutation]

    from sklearn.utils import shuffle
    train = shuffle(train_data)
    s1 = train['s1']
    s2 = train['s2']
    target = train['label']

    # print 'null counts:', target.isnull().sum()
    # print 'train label null counts:', train['label'].isnull().sum()
    # print 's1 null counts:', train['s1'].isnull().sum()
    # print 's2 null counts:', train['s2'].isnull().sum()
    # for t in target.tolist():
    #     print int(t)
    target.map(lambda x: int(x) )

    #!!!!!!!!!!!!!!!!!
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
    all_costs = []

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

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data[0])

    # save model
    eval_acc = round(100 * correct / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; loss {1} mean accuracy {2} :\
              {3}'.format(epoch,round(np.mean(all_costs), 2), eval_type, eval_acc))

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

# Training process

while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1

# Run best model on test set.

nli_net.load_state_dict(ckpt['model'])

print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, 'valid', True)
evaluate(0, 'test', True)


##################### INFERENCE ########################

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


# Inference And Write Result
result = inference(infer_data)
result = pd.DataFrame(result).T
result.to_csv('result.txt',header=False,index=False)

# Save encoder instead of full model
# torch.save(nli_net.encoder.state_dict(), os.path.join(params.outputdir, params.outputmodelname + '.encoder.pkl'))
