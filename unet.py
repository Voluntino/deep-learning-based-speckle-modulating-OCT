from tensorflow.keras.layers import Input,Dense,Conv2D,Flatten,BatchNormalization,UpSampling2D,MaxPooling2D,Activation,Conv2DTranspose,Reshape
from tensorflow.keras.layers import PReLU,Add,Concatenate,LeakyReLU,Dropout,ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.losses import mean_absolute_error as mae
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import cv2
from vggLoss import contentLoss

def createUnet(lr):
    inputLayer = Input(shape = (480, 480, 3))
    conv0 = Conv2D(filters = 32, kernel_size = 1, padding='same')(inputLayer)

    conv1 = Conv2D(filters = 32, kernel_size = 3, padding='same')(conv0)
    conv1 = LeakyReLU()(conv1)
    conv1 = Conv2D(filters = 32, kernel_size = 3, padding='same')(conv1)
    conv1 = LeakyReLU()(conv1)

    max1 = MaxPooling2D(2)(conv1)

    conv2 = Conv2D(filters = 64, kernel_size = 3, padding='same')(max1)
    conv2 = LeakyReLU()(conv2)
    conv2 = Conv2D(filters = 64, kernel_size = 3, padding='same')(conv2)
    conv2 = LeakyReLU()(conv2)

    max2 = MaxPooling2D(2)(conv2)

    conv3 = Conv2D(filters = 128, kernel_size = 3, padding='same')(max2)
    conv3 = LeakyReLU()(conv3)
    conv3 = Conv2D(filters = 128, kernel_size = 3, padding='same')(conv3)
    conv3 = LeakyReLU()(conv3)

    max3 = MaxPooling2D(2)(conv3)

    conv4 = Conv2D(filters = 256, kernel_size = 3, padding='same')(max3)
    conv4 = LeakyReLU()(conv4)
    conv4 = Conv2D(filters = 256, kernel_size = 3, padding='same')(conv4)
    conv4 = LeakyReLU()(conv4)

    max4 = MaxPooling2D(2)(conv4)

    conv5 = Conv2D(filters = 512, kernel_size = 3, padding='same')(max4)
    conv5 = LeakyReLU()(conv5)
    conv5 = Conv2D(filters = 512, kernel_size = 3, padding='same')(conv5)
    conv5 = LeakyReLU()(conv5)
  
    max5 = MaxPooling2D(2)(conv5)

#     max5 = Flatten()(max5)

#     bottle_neck = Dense(512)(max5)
#     bottle_neck = BatchNormalization()(bottle_neck)
#     bottle_neck = Activation('relu')(bottle_neck)
#     bottle_neck = Dense(1024)(bottle_neck)
#     bottle_neck = BatchNormalization()(bottle_neck)
#     bottle_neck = Activation('relu')(bottle_neck)
#     bottle_neck = Reshape((15,15,512))(bottle_neck)

    bottle_neck = Conv2D(filters = 512, kernel_size = 3, padding='same')(max5)
    bottle_neck = LeakyReLU()(bottle_neck)
    bottle_neck = Conv2D(filters = 512, kernel_size = 3, padding='same')(bottle_neck)
    bottle_neck = LeakyReLU()(bottle_neck)
    

    deconv1 = Conv2DTranspose(512, (3,3), strides=(2,2), padding='same')(bottle_neck)
    deconv1 = Concatenate()([conv5, deconv1])
    deconv1 = Conv2D(512, (3,3), padding='same')(deconv1)
    deconv1 = LeakyReLU()(deconv1)
    deconv1 = Conv2D(512, (3,3), padding='same')(deconv1)
    deconv1 = LeakyReLU()(deconv1)

    deconv2 = Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(deconv1)
    deconv2 = Concatenate()([conv4, deconv2])
    deconv2 = Conv2D(256, (3,3), padding='same')(deconv2)
    deconv2 = LeakyReLU()(deconv2)
    deconv2 = Conv2D(256, (3,3), padding='same')(deconv2)
    deconv2 = LeakyReLU()(deconv2)

    deconv3 = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(deconv2)
    deconv3 = Concatenate()([conv3, deconv3])
    deconv3 = Conv2D(128, (3,3), padding='same')(deconv3)
    deconv3 = LeakyReLU()(deconv3)
    deconv3 = Conv2D(128, (3,3), padding='same')(deconv3)
    deconv3 = LeakyReLU()(deconv3)

    deconv4 = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(deconv3)
    deconv4 = Concatenate()([conv2, deconv4])
    deconv4 = Conv2D(64, (3,3), padding='same')(deconv4)
    deconv4 = LeakyReLU()(deconv4)
    deconv4 = Conv2D(64, (3,3), padding='same')(deconv4)
    deconv4 = LeakyReLU()(deconv4)

    deconv5 = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')(deconv4)
    deconv5 = Concatenate()([conv1, deconv5])
    deconv5 = Conv2D(32, (3,3), padding='same')(deconv5)
    deconv5 = LeakyReLU()(deconv5)
    deconv5 = Conv2D(32, (3,3), padding='same')(deconv5)
    deconv5 = LeakyReLU()(deconv5)

    outputLayer = Conv2D(3, (1,1), padding='same')(deconv5)

    model = Model(inputs=inputLayer,outputs=outputLayer)
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss=contentLoss, run_eagerly = True)
    print('-------------------U_net结构---------------------')
    model.summary()

    return model