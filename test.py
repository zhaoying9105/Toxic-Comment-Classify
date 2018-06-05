import pandas as pd
import numpy as np
from keras.models import load_model

# pd.DataFrame(data = )

def test_dict2list():
  dic  = {1:2,3:4,5:6}
  return [[key,value] for key,value in dic.items()]

def test_df(data):
  df = pd.DataFrame(data=data,columns=['id','classes'])
  return df

def test_merge(df1,df2):
  df = pd.merge(df1,df2,on='id')
  return df

def test_pd():
  data = np.array(range(12)).reshape((3,4))
  return pd.DataFrame(data=data)

def func(line):
  print(type(np.array(line)))
  return [line[0] +1 ,line[1] + 1]
  
def test_pd_add_columns():
  df = test_pd()
  df['new'] = df.apply(func,axis=1)
  df['new1'] = df.apply(lambda line:line[4][0],axis=1)
  # del df['new']
  return df

from preprocess import db_init
from preprocess import sentence2vec
def test_precit_for_one():
  epoch=14
  model_path = 'model\\' +'model_'+ str(epoch)+'_'+'.h5'
  net = load_model(model_path)
  col,tokenizer = db_init()
  sent = '''If you have a look back at the source, the information I updated was the correct form. I can only guess the source hadn't updated. I shall update the information once again but thank you for your message.'''
  vec = sentence2vec(col,tokenizer,sent)
  vec = np.array(vec)
  print(vec.shape)
  vec = vec[np.newaxis,:,:]
  result = net.predict_on_batch(vec)
  print(type(result))
  print(result[0])
  print(result[1])

# test_precit_for_one()

def test_add_columns():
  data = [[1,2],[3,4]]
  df = pd.DataFrame(data=data,index=['a','b'])
  df['c'] = [5,6]
  print(df.values)

# test_add_columns()

def test_predict_csv():
  df = pd.read_csv('./predict_new.csv',encoding='utf8')
  # del df['comment_text']
  # df.to_csv('./predict_new.csv',encoding='utf8',index=False)
  print(df.info())

test_predict_csv()
# toxic,severe_toxic,obscene,threat,insult,identity_hate