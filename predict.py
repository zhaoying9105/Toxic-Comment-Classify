from constant import test_data_path
from collections import Counter
import numpy as np
from constant import logger
import tensorflow as tf
from keras.models import load_model
import pandas as pd
from preprocess import sentence2vec
from preprocess import db_init

# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

zero_or_one = lambda x : 0 if x<0.5 else 1
class Line():
  def __init__(self,id,comment,col,tokenizer):
    self.id = id
    self.comment = comment
    self.vec = sentence2vec(col,tokenizer,comment)
  
  def predict(self,net):
    self._class = wrapper_pridict(net,self.vec)[0]

  def to_labels(self):
    self.toxic = zero_or_one(self._class[0])
    self.severe_toxic = zero_or_one(self._class[1])
    self.obscene = zero_or_one(self._class[2])
    self.threat = zero_or_one(self._class[3])
    self.insult = zero_or_one(self._class[4])
    self.identity_hate = zero_or_one(self._class[5])
  def to_data_line(self):
    return [self.id,self.comment,self.toxic,self.severe_toxic,self.obscene,self.threat,self.insult,self.identity_hate]

def test_df(path=test_data_path):
  df = pd.read_csv(path)
  return df

def id_comment_dict(df):
  col,tokenizer = db_init()
  return {item[0]:sentence2vec(col,tokenizer, item[1]) for item in df.values}


epoch = 14

def add1demosion(array):
  if isinstance(array,np.ndarray):
    array = array.tolist()
  # print(len(array.shape))
  # if len(array.shape) == 2:
  #   array = array[np.newaxis,:,:]
  return array

def wrapper_pridict(net,vec):
  if len(vec) == 0:
    return np.array([[0,0,0,0,0,0]])
  if not isinstance(vec,np.ndarray):
    vec = np.array(vec)
  if len(vec.shape) == 2:
    vec =  vec[np.newaxis,:,:]
  return net.predict_on_batch(vec)

def predict():
  logger.info('start predict')
  model_path = 'model\\' +'model_'+ str(epoch)+'_'+'.h5'
  logger.info('local net path is %s', model_path)
  net = load_model(model_path)
  assert net is not None
  logger.info('net loaded')
  df = test_df()
  data = df.values
  col,tokenizer = db_init()
  lines = [Line(item[0],item[1],col,tokenizer) for item in data]
  for line in lines:
    line.predict(net)
  for line in lines:
    line.to_labels()
  new_data = [line.to_data_line() for line in lines]
  new_df = pd.DataFrame(data=new_data,columns=["id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"])
  return new_df

def save_csv(df,path='./predict.csv'):
  df.to_csv(path,encoding='utf-8')
  logger.info('predict result saved to %s',path)

if __name__ == '__main__':
  df = predict()
  save_csv(df)
  # df = test_df()
  # col,tokenizer = db_init()
  # texts = np.array(df['comment_text'])
  # print(len(texts))
  # texts_len = [len(sentence2vec(col,tokenizer,text)) for text in texts]
  # print(dict(Counter(texts_len)))

