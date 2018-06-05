import tensorflow  as tf
import numpy as np
import pandas as pd

from keras.regularizers import l1,l2
from keras.models import Sequential,Model
from keras.layers import Input,Dropout,Concatenate,BatchNormalization
from keras.layers import Dense,Bidirectional,LSTM,GRU
from keras.layers import Conv1D,MaxPool1D

from constant import logger


def Net():
    logger.info('net init...')
    model = Sequential()
    model.add(LSTM(units=25,input_shape=(None,25),dropout=0.1,return_sequences=True))
    model.add(LSTM(units=10,dropout=0.1))
    model.add(Dense(6))
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model

