from tensorflow.keras.layers import Input,Dense,Conv2D,Flatten,BatchNormalization,UpSampling2D,MaxPooling2D,Activation
from tensorflow.keras.layers import PReLU,Add,Concatenate,LeakyReLU,Dropout,ReLU,Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.losses import mean_absolute_error as mae
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import cv2
from vggLoss import contentLoss

def ConvBNReLU(xin, n_filter):
    x = Conv2D(filters = n_filter, kernel_size=3,padding='same')(xin)
    x = BatchNormalization()(x)
    # x = tfa.layers.GroupNormalization()(x)
    x = ReLU()(x)
    return x

def createDncnn(depth, n_filter, lr):
    #输入层
    inputLayer = Input(shape=(None,None,3))
    # Initial conv + relu
    firstLayer = Conv2D(filters=n_filter,kernel_size=3,padding='same')(inputLayer)
    firstLayer = ReLU()(firstLayer)

    middle = firstLayer
    for num in range(depth-2):
        middle = ConvBNReLU(middle, n_filter)
    outputLayer = Conv2D(filters=1,kernel_size=3,padding='same')(middle)
    outputLayer = Subtract()([inputLayer, outputLayer])

    #建模
    model = Model(inputs=inputLayer,outputs=outputLayer)
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss=mse, run_eagerly = True)
    print('-------------------DnCNN结构---------------------')
    model.summary()
    return model
