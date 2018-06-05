from keras import applications

from keras.applications import Xception
from keras.applications import InceptionV3,ResNet50
from keras.layers import Input

a = Input(shape=(300,200,3))
print(type(a))
net = ResNet50(input_tensor=a)
from keras.utils import plot_model
plot_model(net,show_shapes=True,to_file='xception.png')
# print(dir(applications))