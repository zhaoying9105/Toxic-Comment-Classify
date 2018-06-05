from constant import train_data_path,logger
from constant import model_dir,model_name,model_ext
from utils import now_str,calc_sample_weight,total_accuracy
from preprocess import get_train_batch_data_labels,get_valid_batch_data_labels, db_init
from model import Net
import tensorflow as tf
from keras.models import load_model
import numpy as np

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def valid():
  # =====parameters ===========
  col, tokenizer= db_init()
  use_loaded_model =True
  batch_size = 64
  batch_idx = 0
  test_batch_idx = 0

  # ==========currnt==========
  epoch = 14

  # =======net ===============
  net = Net()
  if use_loaded_model:
    try:
      net = load_model(model_dir+model_name+'_'+ str(epoch)+'_'+model_ext)
      logger.info('init net from local, current epoch is %d',epoch)
    except Exception:
        pass
  logger.info('validation start')
  valid_data,valid_labels = get_valid_batch_data_labels(csv_path=train_data_path,col=col,tokenizer=tokenizer,use_batch=False)
  if len(valid_data) > 1:
    logger.info('valid data length is %d',len(valid_data))
    bi = lambda x : 1 if x > 0.5 else 0
    bi_list = lambda label : [ bi(item) for item in label]
    predict_labels = net.predict(valid_data,batch_size=150)
    predict_labels = np.array([bi_list(item) for item in predict_labels])
    logger.info("pridict labels shape: %d,%d",predict_labels.shape[0],predict_labels.shape[1])
    accu = total_accuracy(valid_labels,predict_labels)
    logger.info("accuracy: %f",accu)
  else:
    logger.debug('error!,data in batch_index %d len(data) <= 1',batch_idx)


if __name__ == '__main__':
  valid()