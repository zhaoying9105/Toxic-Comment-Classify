import numpy as np
import tensorflow as tf

from constant import train_data_path,logger
from constant import model_dir,model_name,model_ext
from utils import now_str,calc_sample_weight
from preprocess import get_train_batch_data_labels,get_valid_batch_data_labels, db_init
from model import Net

from keras.models import load_model

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def train():
  # =====parameters ===========
  col, tokenizer= db_init()
  use_loaded_model =True
  batch_size = 64
  batch_idx = 0
  n_epoch = 15


  # ==========currnt==========
  epoch = 1

  # =======net ===============
  net = Net()
  if use_loaded_model:
    try:
      net = load_model(model_dir+model_name+'_'+ str(epoch)+'_'+model_ext)
      logger.info('init net from local, current epoch is %d',epoch)
    except Exception:
        pass

  # =========train ===============
  while epoch < n_epoch:
    logger.info('%s %d / %d epoch start',now_str(),epoch,n_epoch)
    while True:
      if batch_idx % 100 == 0:
        logger.info('%s %d epoch %d batch start',now_str(),epoch,batch_idx)
      data,labels = get_train_batch_data_labels(
        csv_path=train_data_path,col=col,tokenizer=tokenizer,
        batch_index=batch_idx,batch_size=batch_size)
      if len(data) > 1:
        sample_weight = calc_sample_weight(labels)
        net.train_on_batch(data,labels,sample_weight=sample_weight)
      else:
        logger.debug('error!,data in batch_index %d len(data) <= 1',batch_idx)
        break
      batch_idx +=1
    # ===save=====
    logger.info('save model...')
    batch_idx = 0
    net.save(model_dir+model_name+'_'+ str(epoch)+'_'+model_ext)
    epoch+=1


if __name__ == '__main__':
  train()