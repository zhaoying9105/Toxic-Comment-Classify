import sys
sys.path.append('..')
from keras.layers import LSTM,Input
from keras.layers import Permute
from model import Net
from keras.models import Sequential
from keras.utils import plot_model

lstm_net = Net()

plot_model(lstm_net,show_shapes=True, to_file='lstm_model.png')