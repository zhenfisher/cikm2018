#coding=utf-8
import numpy as np
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
stops1 = set(stopwords.words("spanish"))

################### DATA PRE-PROCESSING ####################

df_train_en_sp = pd.read_csv('./input/cikm_english_train_20180516.txt',sep='	', header=None,error_bad_lines=False)
df_train_sp_en = pd.read_csv('./input/cikm_spanish_train_20180516.txt',sep='	', header=None,error_bad_lines=False)
df_train_en_sp.columns = ['english1', 'spanish1', 'english2', 'spanish2', 'result']
df_train_sp_en.columns = ['spanish1', 'english1', 'english2', 'spanish2', 'result']

# Evaluation data
test_data = pd.read_csv('./input/cikm_test_a_20180516.txt', sep='	', header=None,error_bad_lines=False)
test_data.columns = ['spanish1', 'spanish2']

train1 = pd.DataFrame(pd.concat([df_train_en_sp['spanish1'],df_train_sp_en['spanish1']], axis=0))
train2 = pd.DataFrame(pd.concat([df_train_en_sp['spanish2'],df_train_sp_en['spanish2']], axis=0))
train = pd.concat([train1,train2],axis=1).reset_index()
train = train.drop(['index'],axis=1)
result = pd.DataFrame(pd.concat([df_train_en_sp['result'],df_train_sp_en['result']], axis=0)).reset_index()
result = result.drop(['index'],axis=1)
train['result'] = result

# Clean up the weird spanish symbols
# Replacing Spanish capital vowels to lower case

def clean_sent(sent):
    sent = sent.lower()
    sent = re.sub(u'[_"\-;%()|+&=*%.,!?:#$@\[\]/]','',sent)
    sent = re.sub('¡','',sent)
    sent = re.sub('¿','',sent)
    sent = re.sub('Á','á',sent)
    sent = re.sub('Ó','ó',sent)
    sent = re.sub('Ú','ú',sent)
    sent = re.sub('É','é',sent)
    sent = re.sub('Í','í',sent)
    return sent
def cleanSpanish(df):
    df['spanish1'] = df.spanish1.map(lambda x: ' '.join([ word for word in
                                                         nltk.word_tokenize(clean_sent(x).decode('utf-8'))]).encode('utf-8'))
    df['spanish2'] = df.spanish2.map(lambda x: ' '.join([ word for word in
                                                         nltk.word_tokenize(clean_sent(x).decode('utf-8'))]).encode('utf-8'))
def removeSpanishStopWords(df, stop):
    df['spanish1'] = df.spanish1.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))
                                                         if word not in stop]).encode('utf-8'))
    df['spanish2'] = df.spanish2.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))
                                                         if word not in stop]).encode('utf-8'))

cleanSpanish(train)
removeSpanishStopWords(train,stops1)

cleanSpanish(test_data)
removeSpanishStopWords(test_data,stops1)


train_data = train
train_data.columns = ['s1','s2','label']
test_data.columns = ['s1','s2']

#Drop null value data

train_data = train_data.dropna()
test_data = test_data.dropna()

# Save cleaned data

train_data.to_csv("input/cleaned_train.csv", index=False)
test_data.to_csv("input/cleaned_test.csv",index = False)


################### EMBEDDING PRE-PROCESSING ####################

all_sent = train_data['s1'].tolist() + train_data['s2'].tolist() + test_data['s1'].tolist() + test_data['s2'].tolist()

def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1

# Find the number of times each word was used and the size of the vocabulary, word histogram
word_counts = {}
count_words(word_counts, all_sent)
print "Size of Vocabulary:", len(word_counts)

def complete_embedding(embedding_path):
    # create word_vec with embedding vectors
    word_vecs = {}
    with open(embedding_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            word_vecs[word] = np.array(list(map(float, vec.split())))
    return word_vecs

embedding_path = 'input/wiki.es.vec'
complete_embed = complete_embedding(embedding_path)

# find missing words ratios
# Setting a threshold to eliminate the uncommon words, eliminating some junk words

missing_words = []
threshold = 3
for word, count in word_counts.items():
    if count > threshold:
        if word not in complete_embed:
            missing_words.append(word)
missing_ratio = round(float(len(missing_words))/len(word_counts),4)*100

print 'Setting the word frequency threshold to be:', threshold
print "Number of words missing from Embeddings:", len(missing_words)
print "Percent of words that are missing from vocabulary: {}%".format(missing_ratio)


# get vocabularies of significant words

word_dict = {}
value =  0
# for sent in all_sent: ## remove // zhen comment
for word,count in word_counts.items():
    if count > threshold or word in complete_embed:
        word_dict[word] = value
        value +=1

# Special tokens that will be added to our vocab
codes = ["<s>","</s>","<p>","<unk>"]

# Add codes to vocab
for code in codes:
    word_dict[code] = len(word_dict)

print "Total number of unique words:", len(word_counts)
print "Percent of words we will use: {}%".format(round(float(len(word_dict)) / len(word_counts),4)*100)

# Form embedding matrix for our vocab

# def get_embedding(word_dict, embedding_path):
#     # create word_vec with embedding vectors
#     word_vec = {}
#     with open(embedding_path) as f:
#         for line in f:
#             word, vec = line.split(' ', 1)
#             if word in word_dict:
#                 word_vec[word] = np.array(list(map(float, vec.split())))
#     print('Found {0}/{1} words with embedding vectors'.format(
#                 len(word_vec), len(word_dict)))
#     missing_word_num = len(word_dict) - len(word_vec)
#     missing_ratio = round(float(missing_word_num)/len(word_dict),4)*100
#     print('Missing Ratio: {}%'.format(missing_ratio))
#     return word_vec

def get_embedding(word_dict, embedding_path):## speed up
    # create word_vec with embedding vectors
    word_vec = {}
    with open(embedding_path) as f:
        for word in word_dict:
            if word in complete_embed:
                word_vec[word] = complete_embed[word]
    print('Found {0}/{1} words with embedding vectors'.format(
                len(word_vec), len(word_dict)))
    missing_word_num = len(word_dict) - len(word_vec)
    missing_ratio = round(float(missing_word_num)/len(word_dict),4)*100
    print('Missing Ratio: {}%'.format(missing_ratio))
    return word_vec

word_vec = get_embedding(word_dict, embedding_path)

missing = []
for word in word_dict:
    if word not in word_vec:
        missing.append(word)

# save missing words for future use

with open('input/missing_words.csv', 'wb') as fp:
    pickle.dump(missing, fp)


# fill missing words with random vecs
embedding_dim = 300
for word, value in word_dict.items():
    if word not in word_vec:
        # If word not in word_vec, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        word_vec[word] = new_embedding
print "Filled missing words' embeddings."
print "Embedding Matrix Size: ", len(word_vec)

# save embeddings

def save_embed(obj, name ):
    with open('input/'+ name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_embed(word_vec,'embedding.pkl')

print "Embeddings and Data Generated!"