import pandas as pd
# =======logger=============
import logging
logging.basicConfig(filename='./net.log', level=logging.INFO)
logger = logging.getLogger(name="toxi_logger")

# =======data path============
data_dir = 'E:\\data\\kaggle\\Toxic_Comment_Classification'
train_data_path = data_dir + '\\train.csv'
# train_data_path = 'test.csv'
test_data_path = data_dir + '\\test.csv'
sample_path = data_dir + '\\sample_submission.csv'
glove_word2vec_dir = 'E:\\data\\glove-global-vectors-for-word-representation'
word2vec_25d_path = glove_word2vec_dir + '\\glove.twitter.27B.25d.txt'

# =========model path ===============
model_dir = './model\\'
model_name = 'model'
model_ext = '.h5'

train_data_length = 159571

# ========distribution=============
label_list = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
label_cls_dict = {'threat': 478, 'total': 159571, 'severe_toxic': 1595, 'normal': 143346, 'insult': 7877, 'obscene': 8449, 'toxic': 15294, 'identity_hate': 1405}
label_cls_ratio_dict = {'threat': 0.002995531769557125, 'severe_toxic': 0.009995550569965721, 'insult': 0.04936360616904074, 'total': 1.0, 'identity_hate': 0.00880485802558109, 'obscene': 0.052948217407925, 'normal': 0.8983211235124177, 'toxic': 0.09584448302009764}
label_num_dict = {'2': 3480, '6': 31, '4': 1760, '3': 4209, '1': 6360, 'total': 159571, '0': 143346, '5': 385}
label_num_ratio_dict = {'2': 0.0218084739708343, '5': 0.002412719103095174, '4': 0.011029573042720795, '3': 0.026376973259552173, '1': 0.03985686622255924, 'total': 1.0, '0': 0.8983211235124177, '6': 0.00019427088882065038}

# =========anaylsis to get distribution ==================
def analysis(train_data_path):
  train_data = pd.read_csv(train_data_path)
  # print(train_data.columns)
  total_size = train_data.shape[0]
  toxic_count = train_data.loc[train_data.toxic > 0].shape[0]
  severe_toxic_count = train_data.loc[train_data.severe_toxic > 0].shape[0]
  obscene_count = train_data.loc[train_data.obscene > 0].shape[0]
  threat_count = train_data.loc[train_data.threat > 0].shape[0]
  insult_count = train_data.loc[train_data.insult > 0].shape[0]
  identity_hate_count = train_data.loc[train_data.identity_hate > 0].shape[0]
  normal_count = train_data.loc[train_data.toxic+train_data.severe_toxic+train_data.obscene+train_data.threat+train_data.insult+train_data.identity_hate == 0].shape[0]

  cls_dict = {
    'toxic':toxic_count,
    'severe_toxic':severe_toxic_count,
    'obscene':obscene_count,
    'threat':threat_count,
    'insult':insult_count,
    'identity_hate':identity_hate_count,
    'normal':normal_count,
    'total':total_size
  }

  mutil_0_count = train_data.loc[train_data.toxic+train_data.severe_toxic+train_data.obscene+train_data.threat+train_data.insult+train_data.identity_hate == 0].shape[0]  
  mutil_1_count = train_data.loc[train_data.toxic+train_data.severe_toxic+train_data.obscene+train_data.threat+train_data.insult+train_data.identity_hate == 1].shape[0]
  mutil_2_count = train_data.loc[train_data.toxic+train_data.severe_toxic+train_data.obscene+train_data.threat+train_data.insult+train_data.identity_hate == 2].shape[0]
  mutil_3_count = train_data.loc[train_data.toxic+train_data.severe_toxic+train_data.obscene+train_data.threat+train_data.insult+train_data.identity_hate == 3].shape[0]
  mutil_4_count = train_data.loc[train_data.toxic+train_data.severe_toxic+train_data.obscene+train_data.threat+train_data.insult+train_data.identity_hate == 4].shape[0]
  mutil_5_count = train_data.loc[train_data.toxic+train_data.severe_toxic+train_data.obscene+train_data.threat+train_data.insult+train_data.identity_hate == 5].shape[0]
  mutil_6_count = train_data.loc[train_data.toxic+train_data.severe_toxic+train_data.obscene+train_data.threat+train_data.insult+train_data.identity_hate == 6].shape[0]

  multi_cls_dict = {
    '0':mutil_0_count,
    '1': mutil_1_count,
    '2': mutil_2_count,
    '3': mutil_3_count,
    '4': mutil_4_count,
    '5': mutil_5_count,
    '6': mutil_6_count,
    'total' : total_size
  }

  print(cls_dict)
  print({key:value / total_size for key,value in cls_dict.items()})
  print(multi_cls_dict)
  print({key:value / total_size for key,value in multi_cls_dict.items()})


# ========show distribution in a pie picture ===================
# def show_pie():
  # bad_counts = [toxic_count,severe_toxic_count,obscene_count,threat_count,insult_count,identity_hate_count]
  # total_bad_count = np.sum(np.array(bad_counts))
  # from matplotlib import pyplot as plt
  # labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','normal']
  # fracs = [count / total_size for count in bad_counts]
  # fracs.append((total_size -total_bad_count)/total_size)
  # plt.pie(fracs,labels=labels,shadow=True)
  # plt.savefig('pie_cls.png')
