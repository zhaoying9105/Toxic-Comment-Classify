from keras.engine.topology import Layer
import numpy as np

class Mylayer(Layer):

  def __init__(self,output_dim):
    self.output_dim = output_dim
  
  def build(self,input_shape):
    self.kernel = self.add_weight(name='my_layer')