from pymongo import MongoClient
import pandas as pd
import sys
import numpy as np
import io
import re
from constant import train_data_path,train_data_path,train_data_length,logger
from nltk.tokenize import WordPunctTokenizer
from keras.preprocessing import sequence

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码

# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46371
def substitute_repeats_fixed_len(text, nchars, ntimes=3):
    return re.sub(r"(\S{{{}}})(\1{{{},}})".format(nchars, ntimes-1), r"\1", text)

def substitute_repeats(text, ntimes=3):
    for nchars in range(1, 20):
        text = substitute_repeats_fixed_len(text, nchars, ntimes)
    return text

def word2vec(col,word):
  try:
    return col.find_one({'_id':word})['vector']
  except Exception as ex:
    logger.debug('%s cant be found.'% word)
    return None

# 单词列表转换成向量
def words2vecs(col,words):
  condition = {
    '_id': {
      '$in': words
    },
  }
  documents = list(col.find(condition))
  vec_dict = { document['_id']: document['vector'] for document in documents }
  vecs = []
  for word in words:
    try:
      vec = vec_dict[word]
      vecs.append(vec)
    except KeyError:
      pass
  return vecs

# 句子拆成单词列表，可以设置数量上限
def sentence2words(tokenizer,sentence,is_limit=True ,len_limit = 150):
  words = tokenizer.tokenize(sentence)
  filter_words = [word.lower() for word in words if not re.match(r'\d+|[\$\%\&\*\.,;"\'?\(\)\:\-\_\`]',word)]
  if is_limit and len(filter_words) > len_limit:
    return filter_words[:len_limit]
  else:
    return filter_words

def get_part_data_labels(csv_path,start=0,nrows=None):
  try:
    if nrows == None:
      raw_data = pd.read_csv(csv_path,skiprows=start)
    else:
      raw_data = pd.read_csv(csv_path,skiprows=start,nrows=nrows)
    return raw_data.iloc[:,1].values,raw_data.iloc[:,2:8].values
  except Exception as err:
    return [],[]

def sentence2vec(col,tokenizer,sent):
  words = sentence2words(tokenizer,sent)
  vecs = words2vecs(col,words)
  return vecs

def get_valid_batch_data_labels(csv_path,col,tokenizer,use_batch=True,batch_index=0,batch_size=64,split_ratio=0.9,data_len=train_data_length):
  if use_batch:
    if (batch_index + 1)* batch_size < train_data_length * (1 - split_ratio):
      comments, labels = get_part_data_labels(csv_path,start = int(train_data_length * split_ratio + batch_index * batch_size),nrows =batch_size)
      data = [sentence2vec(col,tokenizer,sent) for sent in comments]
      if None in data :
        logger.warning('data is none')
        print('data is none')
      data = sequence.pad_sequences(data,padding='post',dtype='float32',value=0.)
      return data, labels
    else:
      return [],[]
  else:
    comments, labels = get_part_data_labels(csv_path,start =int(train_data_length * split_ratio+ batch_index * batch_size))
    if len(comments) == 0:
      logger.error('valid data is empty')
    data = [sentence2vec(col,tokenizer,sent) for sent in comments]
    data = sequence.pad_sequences(data,padding='post',dtype='float32',value=0.)
    return data, labels
def get_train_batch_data_labels(csv_path,col,tokenizer,batch_index=0,batch_size=64,is_split=True,split_ratio=0.9,data_len=train_data_length):
  if is_split == True:
    if (batch_index +1) * batch_size < train_data_length * split_ratio:
      # logger.debug('get data start= %d,size =%d',batch_index * batch_size,batch_size)
      comments, labels = get_part_data_labels(csv_path,start =batch_index * batch_size,nrows =batch_size)
      # print("label shape: ",labels.shape)
      if len(comments) ==0 and len(labels) ==0:
        logger.debug('get_train_batch_data_labels return empty list for get_part_data_labels is empty')
        return [],[]
      data = [sentence2vec(col,tokenizer,sent) for sent in comments]
      if None in data :
        logger.warning('data is none')
        print('none count :',data.count(None))
      data = sequence.pad_sequences(data,padding='post',dtype='float32',value=0.)
      # print(data.shape)
      return data, labels
    else:
      logger.debug('get_train_batch_data_labels return empty list for  (batch_index +1) * batch_size > train_data_length * split_ratio')
      return [],[]
  else:
    comments, labels = get_part_data_labels(csv_path,start =batch_index * batch_size,nrows =batch_size)
    data = [sentence2vec(col,tokenizer,sent) for sent in comments]
    data = sequence.pad_sequences(data,padding='post',value=0.)
    return data,labels

def db_init():
  logger.info('database connect....')
  client = MongoClient('mongodb://localhost:27017')
  db = client.word2vec_twitter_25d
  col = db['word2vec']
  tokenizer = WordPunctTokenizer()
  return col,tokenizer
def main():
  col, tokenizer = db_init()
  sent = "We'll use cookies to enhance, your experience on our website".lower()
  vecs = sentence2vec(col,tokenizer,sent)
  print(vecs)

def main2():
  sents,label = get_part_data_labels(train_data_path,start=0,nrows=3)
  print(sents)
  print(label)

def test_word_reg():
  sent = "Explanation\nWhy the edits made under my () 2to3 username 16 in 2016 Hardcore Metallica Fan were reverted?"
  tokenizer = WordPunctTokenizer()
  print(sentence2words(tokenizer,sent))


# if __name__ == '__main__':
    # col, tokenizer = db_init()
    # sent = "Explanation\nWhy the edits made under my () 2to3 username 16 in 2016 Hardcore Metallica Fan were reverted?"
    # words = sentence2words(tokenizer,sent)
    # print(words2vecs(col=col,words=words))