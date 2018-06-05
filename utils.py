import time
import numpy as np
import pandas as pd
from collections import Counter
from constant import train_data_path,label_cls_ratio_dict,label_list
from constant import logger
from matplotlib import pyplot as plt

# 指定字符串显示当前时间
def now_str(format="%Y-%m-%d %X"):
  return time.strftime(format, time.localtime())

# 数据集中数据量
def csv_file_length(csv_file_path):
  # 训练数据：1276568
  df = pd.read_csv(csv_file_path)
  return len(df)

# 数据集中句子长度分布
def get_sent_length_distribution(csv_file_path):
  raw_data = pd.read_csv(csv_file_path)
  sents = raw_data['comment_text'].values
  lens = [len(sent) for sent in sents]
  counter = dict(Counter(lens))
  return counter

# 折线图表示数据集中句子长度分布
def show_sent_length_distribution():
  distribution = get_sent_length_distribution(train_data_path)
  x = list(distribution.keys())
  y = list(distribution.values())
  plt.xlabel('len of sentence')
  plt.ylabel('counter')
  plt.plot(x,y)
  plt.show()

# show_sent_length_distribution()

def calc_sample_weight(labels):
  weights = np.array([calc_single_weight(label) for label in labels])
  total = np.sum(weights)
  return weights / total
  
def calc_single_weight(label):
  weight = 0
  for i,value in enumerate(label):
    if label_list[i] == 1:
      weight += 1 / label_cls_ratio_dict[label_list[i]]
    else:
      weight +=1 / (1-label_cls_ratio_dict[label_list[i]])
  return weight


def total_accuracy(labels,predicts):
  if len(labels) > len(predicts):
    logger.warning('predict labels length < origin labels')
    labels = labels[:len(predicts)]
  assert len(labels) == len(predicts)
  result = np.array([ (label == predict).all() for label,predict in zip(labels,predicts) ])
  accu = np.sum(result) / len(labels)
  return accu
