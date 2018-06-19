#coding:utf-8

import sys 
reload(sys) 
sys.setdefaultencoding('utf-8')
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import xgboost as xgb

from featwheel.utils import NgramUtil,DistanceUtil, LogUtil, MathUtil

from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('spanish')
snowball_stemmer.stem

from nltk.corpus import stopwords
stops = set(stopwords.words("spanish"))


from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.cross_validation import cross_val_score


print '===read data==='
df_train_en_sp = pd.read_csv('./input/cikm_english_train_20180516.txt',sep='	', header=None,error_bad_lines=False)
df_train_sp_en = pd.read_csv('./input/cikm_spanish_train_20180516.txt',sep='	', header=None,error_bad_lines=False)
df_train_en_sp.columns = ['english1', 'spanish1', 'english2', 'spanish2', 'result']
df_train_sp_en.columns = ['spanish1', 'english1', 'english2', 'spanish2', 'result']

test_data = pd.read_csv('./input/cikm_test_a_20180516.txt', sep='	', header=None,error_bad_lines=False)
test_data.columns = ['spanish1', 'spanish2']

print "===clean text==="
def subchar(text):
	text=text.replace("á", "a")
	text=text.replace("ó", "o")
	text=text.replace("é", "e")	
	text=text.replace("í", "i")
	text=text.replace("ú", "u")	
	return text

def cleanSpanish(df):
	df['spanish1'] = df.spanish1.map(lambda x: ' '.join([snowball_stemmer.stem(word) for word in nltk.word_tokenize(x.lower().decode('utf-8'))]).encode('utf-8'))
	df['spanish2'] = df.spanish2.map(lambda x: ' '.join([snowball_stemmer.stem(word) for word in nltk.word_tokenize(x.lower().decode('utf-8'))]).encode('utf-8'))
	#西班牙语缩写还原#




def removestopwords(df, stop):
	df['spanish1'] = df.spanish1.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))  if word not in stop]).encode('utf-8'))
	df['spanish2'] = df.spanish2.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))  if word not in stop]).encode('utf-8'))

cleanSpanish(df_train_en_sp)
cleanSpanish(df_train_sp_en)


train_data1 = df_train_en_sp.drop(['english1','english2'],axis=1)
train_data2 = df_train_sp_en.drop(['english1','english2'],axis=1)

train_data = pd.concat([train_data1,train_data2],axis=0)

print test_data.shape

cleanSpanish(test_data)


stptrain = train_data.copy()
stptest = test_data.copy()
removestopwords(stptrain, stops)
removestopwords(stptest, stops)


print "===feature engineering==="
#negative words count
def negativeWordsCount(row):
	q1 = nltk.word_tokenize(row['spanish1'])
	q2 = nltk.word_tokenize(row['spanish2'])

	Ncount1 = 0
	Ncount2 = 0
	
	negw = ['no','nadie','nada','nunca','jamás','ningún','ninguno','ninguna','ningunos','ningunas']

	for wd in negw:
		Ncount1 += q1.count(wd)
		Ncount2 += q2.count(wd)
	
	fs = list()
        fs.append(Ncount1)
        fs.append(Ncount2)

	if Ncount1 == Ncount2:
		if Ncount1 == 0:
            		fs.append(0.)
        	else:
            		fs.append(1.)
	else:
		if Ncount1 == 0 or Ncount2 == 0:
			fs.append(2.)
		else:
			fs.append(3.)

	return fs


 
def shareWordscnt(row):
	fs=list()
	q1words = {}
        q2words = {}
        for word in str(row['spanish1']).lower().split():
            if word not in stops:
                q1words[word] = q1words.get(word, 0) + 1
        for word in str(row['spanish2']).lower().split():
            if word not in stops:
                q2words[word] = q2words.get(word, 0) + 1
        n_shared_word_in_q1 = sum([q1words[w] for w in q1words if w in q2words])
        n_shared_word_in_q2 = sum([q2words[w] for w in q2words if w in q1words])
        n_tol = sum(q1words.values()) + sum(q2words.values())
        if 1e-6 > n_tol:
            fs.append(0.)
        else:
            fs.append(1.0 * (n_shared_word_in_q1 + n_shared_word_in_q2) / n_tol)
	return fs




def init_idf(data):
        idf = {}
        q_set = set()
        for index, row in data.iterrows():
            q1 = str(row['spanish1'])
            q2 = str(row['spanish2'])
            if q1 not in q_set:
                q_set.add(q1)
                words = q1.lower().split()
                for word in words:
                    idf[word] = idf.get(word, 0) + 1
            if q2 not in q_set:
                q_set.add(q2)
                words = q2.lower().split()
                for word in words:
                    idf[word] = idf.get(word, 0) + 1
        num_docs = len(data)
        for word in idf:
            idf[word] = math.log(num_docs / (idf[word] + 1.)) / math.log(2.)
        LogUtil.log("INFO", "idf calculation done, len(idf)=%d" % len(idf))
        return idf




def lengthDiff(row):
        q1 = str(row['spanish1'])
        q2 = str(row['spanish2'])
	l1=len(q1)
	l2=len(q2)

	sl1=len(q1.split())
	sl2=len(q2.split())


        fs = list()
        fs.append(l1)
        fs.append(l2)
        fs.append(sl1)
        fs.append(sl2)
	fs.append(abs(l1 - l2))
	fs.append(abs(sl1 - sl2))
	
	if max(l1, l2) < 1e-6:
            fs.append(0.)
        else:
            fs.append(1.0 * min(l1, l2) / max(l1, l2))

	if max(sl1, sl2) < 1e-6:
        	fs.append(0.)
        else:
        	fs.append(1.0 * min(sl1, sl2) / max(sl1, sl2))
	
	return fs



def generate_powerful_word(data):
        """
        计算数据中词语的影响力，格式如下：
            词语 --> [0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
        """
	subset_indexs = data.shape[0]
        words_power = {}
        for index, row in data.iterrows():
            label = int(row['result'])
            q1_words = str(row['spanish1']).lower().split()
            q2_words = str(row['spanish2']).lower().split()
            all_words = set(q1_words + q2_words)
            q1_words = set(q1_words)
            q2_words = set(q2_words)
            for word in all_words:
                if word not in words_power:
                    words_power[word] = [0. for i in range(7)]
                # 计算出现语句对数量
                words_power[word][0] += 1.
                words_power[word][1] += 1.

                if ((word in q1_words) and (word not in q2_words)) or ((word not in q1_words) and (word in q2_words)):
                    # 计算单侧语句数量
                    words_power[word][3] += 1.
                    if 0 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算单侧语句正确比例
                        words_power[word][4] += 1.
                if (word in q1_words) and (word in q2_words):
                    # 计算双侧语句数量
                    words_power[word][5] += 1.
                    if 1 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算双侧语句正确比例
                        words_power[word][6] += 1.
        for word in words_power:
            # 计算出现语句对比例
            words_power[word][1] /= subset_indexs
            # 计算正确语句对比例
            words_power[word][2] /= words_power[word][0]
            # 计算单侧语句对正确比例
            if words_power[word][3] > 1e-6:
                words_power[word][4] /= words_power[word][3]
            # 计算单侧语句对比例
            words_power[word][3] /= words_power[word][0]
            # 计算双侧语句对正确比例
            if words_power[word][5] > 1e-6:
                words_power[word][6] /= words_power[word][5]
            # 计算双侧语句对比例
            words_power[word][5] /= words_power[word][0]
        sorted_words_power = sorted(words_power.iteritems(), key=lambda d: d[1][0], reverse=True)
        LogUtil.log("INFO", "power words calculation done, len(words_power)=%d" % len(sorted_words_power))
        return sorted_words_power

def save_powerful_word(words_power, fp):
        f = open(fp, 'w')
        for ele in words_power:
            f.write("%s" % ele[0])
            for num in ele[1]:
                f.write("\t%.5f" % num)
            f.write("\n")
        f.close()

def load_powerful_word(fp):
        powful_word = []
        f = open(fp, 'r')
        for line in f:
            subs = line.split('\t')
            word = subs[0]
            stats = [float(num) for num in subs[1:]]
            powful_word.append((word, stats))
        f.close()
        return powful_word


def init_powerful_word_dside(pword, thresh_num, thresh_rate):
        pword_dside = []
        pword = filter(lambda x: x[1][0] * x[1][5] >= thresh_num, pword)
        pword_sort = sorted(pword, key=lambda d: d[1][6], reverse=True)
        pword_dside.extend(map(lambda x: x[0], filter(lambda x: x[1][6] >= thresh_rate, pword_sort)))
        LogUtil.log('INFO', 'Double side power words(%d): %s' % (len(pword_dside), str(pword_dside)))
        return pword_dside



def extract_dside(pword_dside, row):
        fs = []
        q1_words = str(row['spanish1']).lower().split()
        q2_words = str(row['spanish2']).lower().split()
        for word in pword_dside:
            if (word in q1_words) and (word in q2_words):
                fs.append(1.0)
            else:
                fs.append(0.0)
        return fs



def init_powerful_word_oside(pword, thresh_num, thresh_rate):
        pword_oside = []
        pword = filter(lambda x: x[1][0] * x[1][3] >= thresh_num, pword)
        pword_oside.extend(
            map(lambda x: x[0], filter(lambda x: x[1][4] >= thresh_rate, pword)))
        LogUtil.log('INFO', 'One side power words(%d): %s' % (
            len(pword_oside), str(pword_oside)))
        return pword_oside

def extract_oside(pword_oside, row):
        fs = []
        q1_words = set(str(row['spanish1']).lower().split())
        q2_words = set(str(row['spanish2']).lower().split())
        for word in pword_oside:
            if (word in q1_words) and (word not in q2_words):
                fs.append(1.0)
            elif (word not in q1_words) and (word in q2_words):
                fs.append(1.0)
            else:
                fs.append(0.0)
        return fs


def extract_dside_rate(pword, row):
	pword_dict = dict(pword)
        num_least = 300
        rate = [1.0]
        q1_words = set(str(row['spanish1']).lower().split())
        q2_words = set(str(row['spanish2']).lower().split())
        share_words = list(q1_words.intersection(q2_words))
        for word in share_words:
            if word not in pword_dict:
                continue
            if pword_dict[word][0] * pword_dict[word][5] < num_least:
                continue
            rate[0] *= (1.0 - pword_dict[word][6])
        rate = [1 - num for num in rate]
        return rate

def extract_oside_rate(pword, row):
        pword_dict = dict(pword)
	num_least = 300
        rate = [1.0]
        q1_words = set(str(row['spanish1']).lower().split())
        q2_words = set(str(row['spanish2']).lower().split())
        q1_diff = list(set(q1_words).difference(set(q2_words)))
        q2_diff = list(set(q2_words).difference(set(q1_words)))
        all_diff = set(q1_diff + q2_diff)
        for word in all_diff:
            if word not in pword_dict:
                continue
            if pword_dict[word][0] * pword_dict[word][3] < num_least:
                continue
            rate[0] *= (1.0 - pword_dict[word][4])
        rate = [1 - num for num in rate]
        return rate



def init_tfidf(train_data,test_data):
        tfidf = TfidfVectorizer(ngram_range=(1, 1))
        tfidf_txt = pd.Series(train_data['spanish1'].tolist() + train_data['spanish2'].tolist() + test_data['spanish1'].tolist() + test_data['spanish2'].tolist()).astype(str)
        tfidf.fit_transform(tfidf_txt)
        LogUtil.log("INFO", "init tfidf done ")
        return tfidf

def extract_tfidf(row,tfidf):
        q1 = str(row['spanish1'])
        q2 = str(row['spanish2'])

        fs = list()
        fs.append(np.sum(tfidf.transform([str(q1)]).data))
        fs.append(np.sum(tfidf.transform([str(q2)]).data))
        fs.append(np.mean(tfidf.transform([str(q1)]).data))
        fs.append(np.mean(tfidf.transform([str(q2)]).data))
        fs.append(len(tfidf.transform([str(q1)]).data))
        fs.append(len(tfidf.transform([str(q2)]).data))
        return fs



def generate_dul_num(train_data,test_data):
        dul_num = {}
        for index, row in train_data.iterrows():
            q1 = str(row.spanish1).strip()
            q2 = str(row.spanish2).strip()
            dul_num[q1] = dul_num.get(q1, 0) + 1
            if q1 != q2:
                dul_num[q2] = dul_num.get(q2, 0) + 1
        for index, row in test_data.iterrows():
            q1 = str(row.spanish1).strip()
            q2 = str(row.spanish2).strip()
            dul_num[q1] = dul_num.get(q1, 0) + 1
            if q1 != q2:
                dul_num[q2] = dul_num.get(q2, 0) + 1
        return dul_num


def extract_dul_fs(dul_num, row):
        q1 = str(row['spanish1']).strip()
        q2 = str(row['spanish2']).strip()

        dn1 = dul_num[q1]
        dn2 = dul_num[q2]
        return [dn1, dn2, max(dn1, dn2), min(dn1, dn2)]


def extract_math(row):
        q1 = str(row['spanish1']).strip()
        q2 = str(row['spanish2']).strip()

        q1_cnt = q1.count('[math]')
        q2_cnt = q2.count('[math]')
        pair_and = int((0 < q1_cnt) and (0 < q2_cnt))
        pair_or = int((0 < q1_cnt) or (0 < q2_cnt))
        return [q1_cnt, q2_cnt, pair_and, pair_or]



def extract_diff_char( row):
        s = 'abcdefghijklmnopqrstuvwxyz'
        q1 = str(row['spanish1']).strip()
        q2 = str(row['spanish2']).strip()
	q1 = subchar(q1)
	q2 = subchar(q2)
	
        fs1 = [0] * 26
        fs2 = [0] * 26
        for index in range(len(q1)):
            c = q1[index]
            if 0 <= s.find(c):
                fs1[s.find(c)] += 1
        for index in range(len(q2)):
            c = q2[index]
            if 0 <= s.find(c):
                fs2[s.find(c)] += 1
        return [np.sum(abs(np.array(fs1) - np.array(fs2)))]

def extract_ngram_jaccard_distance(row):
        q1_words = str(row['spanish1']).lower().split()
        q2_words = str(row['spanish2']).lower().split()
        fs = []
        for n in range(1, 4):
            q1_ngrams = NgramUtil.ngrams(q1_words, n)
            q2_ngrams = NgramUtil.ngrams(q2_words, n)
            fs.append(DistanceUtil.jaccard_coef(q1_ngrams, q2_ngrams))
        return fs

def extract__ngram_dice_distance(row):
        q1_words = str(row['spanish1']).lower().split()
        q2_words = str(row['spanish2']).lower().split()
        fs = []
        for n in range(1, 4):
            q1_ngrams = NgramUtil.ngrams(q1_words, n)
            q2_ngrams = NgramUtil.ngrams(q2_words, n)
            fs.append(DistanceUtil.dice_dist(q1_ngrams, q2_ngrams))
        return fs

def extract_edt_cp_distance(row):
	fs = list()
        q1 = str(row['spanish1']).strip()
        q2 = str(row['spanish2']).strip()
	fs.append(DistanceUtil.edit_dist(q1, q2))
	fs.append(DistanceUtil.compression_dist(q1, q2))        
	return fs

def extract_raw_adt_cp_distance(raw_row):
         q1 = str(raw_row['spanish1']).strip()
         q2 = str(raw_row['spanish2']).strip()
         fs.append(DistanceUtil.edit_dist(q1, q2))
         fs.append(DistanceUtil.compression_dist(q1, q2))
         return fs         



def stat_feature_gen(train_data,test_data,tfidfx):

	pword = generate_powerful_word(train_data)

	thresh_num=500
	thresh_rate=0.9

	pword_dside = init_powerful_word_dside(pword, thresh_num, thresh_rate)
	pword_oside = init_powerful_word_oside(pword, thresh_num, thresh_rate)
	dul_num = generate_dul_num(train_data,test_data)

	all_data = pd.concat([train_data,test_data],axis=0)	
	nlist = range(1,38)
	dfs = pd.DataFrame(columns=nlist)#('notc','shc','ld','pd','po','pdr','por','tf1','tf2','tf3','tf4','tf5','tf6','dul1','dul2','dul3','dul4','m1','m2','m3','m4','char','jar','dd','ed','1','2','3','4','5',''))
	i = 0
	for index, row in all_data.iterrows(): 
		fs = []
		fs += negativeWordsCount(row)
		fs += shareWordscnt(row)
		fs += lengthDiff(row)
		fs += extract_dside(pword_dside, row)
		fs += extract_oside(pword_oside,row)
		fs += extract_dside_rate(pword, row)
		fs += extract_oside_rate(pword, row)		
		fs += extract_tfidf(row,tfidfx) 
		fs += extract_dul_fs(dul_num,row) 
		fs += extract_math(row) 
		fs += extract_diff_char(row)  
		fs += extract_ngram_jaccard_distance(row) 
		fs += extract__ngram_dice_distance(row) 
		#fs += extract_edt_cp_distance(row)
#	for index, row in raw_data.iterrows():
#		fsredt = extract_raw_edt_cp_distance(row)
		dfs.loc[i] = fs
		i += 1
	print i
	return dfs


#tfidfx = init_tfidf(stptrain,stptest)
#statcfs = stat_feature_gen(train_data,test_data,tfidfx)
#statcfs.to_csv('statfs.csv')

statcfs = pd.read_csv('./statfs.csv')
del statcfs['Unnamed: 0']
#print statcfs.shape
#print statcfs.head()
y_train = train_data['result']
#print y_train.shape


traindata = statcfs[:train_data.shape[0]]
testdata = statcfs[train_data.shape[0]:]

print testdata.shape


xgbst = xgb.XGBClassifier(nthread=4, learning_rate=0.08,n_estimators=3000, max_depth=4, gamma=0, subsample=0.9, colsample_bytree=0.5)
print -cross_val_score(xgbst,traindata, y_train, cv = 5,scoring = 'neg_log_loss').mean()

xgbst.fit(traindata, y_train)
testdata['predicted_score'] = xgbst.predict_proba(testdata)[:, 1]
testdata[[ 'predicted_score']].to_csv('submity.txt', index=False)
print("...........end xgboost.........")






















	




















	


	














