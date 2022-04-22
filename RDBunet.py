from tensorflow.keras.layers import Input,Dense,Conv2D,Flatten,BatchNormalization,UpSampling2D,MaxPooling2D,Activation,Conv2DTranspose,Reshape
from tensorflow.keras.layers import PReLU,Add,Concatenate,LeakyReLU,Dropout,ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.losses import mean_absolute_error as mae
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from vggLoss import contentLoss
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import cv2

def RDB(xin, n_filters):
    x1 = Conv2D(filters = n_filters, kernel_size = 3, padding='same')(xin)
    x1 = LeakyReLU()(x1)
    x1 = Concatenate()([xin, x1])
    x2 = Conv2D(filters = n_filters, kernel_size = 3, padding='same')(x1)
    x2 = LeakyReLU()(x2)
    x2 = Concatenate()([x1, x2])
    x3 = Conv2D(filters = n_filters, kernel_size = 3, padding='same')(x2)
    x3 = LeakyReLU()(x3)
    x3 = Concatenate()([x2, x3])

    x4 = Conv2D(filters = n_filters, kernel_size = 1, padding='same')(x3)
    x4 =Add()([xin, x4])

    return x4


def createRDBUnet(lr):
    inputLayer = Input(shape = (None, None, 3))
    conv1 = Conv2D(filters = 32, kernel_size = 1, padding='same')(inputLayer)

    rdb1 = RDB(conv1, 32)
    rdb11 = RDB(rdb1, 32)

    max1 = MaxPooling2D(2)(rdb1)
    max1 = Conv2D(filters = 64, kernel_size = 1, padding='same')(max1)

    rdb2 = RDB(max1, 64)
    rdb22 = RDB(rdb2, 64)

    max2 = MaxPooling2D(2)(rdb2)
    max2 = Conv2D(filters = 128, kernel_size = 1, padding='same')(max2)

    rdb3 = RDB(max2, 128)
    rdb33 = RDB(rdb3, 128)

    max3 = MaxPooling2D(2)(rdb3)
    max3 = Conv2D(filters = 256, kernel_size = 1, padding='same')(max3)

    rdb4 = RDB(max3, 256)
    rdb44 = RDB(rdb4, 256)

    max4 = MaxPooling2D(2)(rdb4)
    max4 = Conv2D(filters = 512, kernel_size = 1, padding='same')(max4)

    rdb5 = RDB(max4, 512)
    rdb55 = RDB(rdb5, 512)

    max5 = MaxPooling2D(2)(rdb5)
    max5 = Conv2D(filters = 1024, kernel_size = 1, padding='same')(max5)

    ###
    bottle_neck = RDB(max5, 1024)
    ###

    deconv1 = Conv2DTranspose(512, (3,3), strides=(2,2), padding='same')(bottle_neck)
    deconv1 = Concatenate()([rdb55, deconv1])
    derdb1 = RDB(deconv1, 1024)
    derdb1 = Conv2D(filters = 512, kernel_size = 1, padding='same')(derdb1)

    deconv2 = Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(derdb1)
    deconv2 = Concatenate()([rdb44, deconv2])
    derdb2 = RDB(deconv2, 512)
    derdb2 = Conv2D(filters = 256, kernel_size = 1, padding='same')(derdb2)

    deconv3 = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(derdb2)
    deconv3 = Concatenate()([rdb33, deconv3])
    derdb3 = RDB(deconv3, 256)
    derdb3 = Conv2D(filters = 128, kernel_size = 1, padding='same')(derdb3)

    deconv4 = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(derdb3)
    deconv4 = Concatenate()([rdb22, deconv4])
    derdb4 = RDB(deconv4, 128)
    derdb4 = Conv2D(filters = 64, kernel_size = 1, padding='same')(derdb4)

    deconv5 = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')(derdb4)
    deconv5 = Concatenate()([rdb11, deconv5])
    derdb5 = RDB(deconv5, 64)
    derdb5 = Conv2D(filters = 32, kernel_size = 1, padding='same')(derdb5)

    outputLayer = Conv2D(3, (1,1), padding='same')(derdb5)
#     outputLayer = BatchNormalization()(outputLayer)
#     outputLayer = Activation('sigmoid')(outputLayer)

    model = Model(inputs=inputLayer,outputs=outputLayer)
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss=contentLoss, run_eagerly = True)
    print('-------------------U_net结构---------------------')
    model.summary()

    return model