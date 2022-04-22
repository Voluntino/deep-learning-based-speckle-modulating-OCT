#!/usr/bin/env python
# coding: utf-8

# In[13]:


from tensorflow.keras.layers import Input,Dense,Conv2D,Flatten,BatchNormalization,UpSampling2D,MaxPooling2D,Activation
from tensorflow.keras.layers import PReLU,Add,Concatenate,LeakyReLU,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.losses import mean_absolute_error as mae
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow.keras.backend as K
from vggLoss import contentLoss
import tensorflow as tf
import numpy as np
import cv2


# def resBlock(xIn,filterNum):
#     '''残差块'''
#     x = Conv2D(filters=filterNum,kernel_size=3,padding='same')(xIn)
# #     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)
#     x = Conv2D(filters=filterNum,kernel_size=3,padding='same')(x)
# #     x = BatchNormalization()(x)
# #     x = LeakyReLU()(x)
#     x = Add()([xIn, x])
#     return x

def ResDenseLayer(x, nb_filter, bn_size=4):
    x = Conv2D(filters = nb_filter, kernel_size=3,padding='same')(x)
    x = LeakyReLU()(x)
    # 这里没有加上Dropout()层，后面看效果再考虑

def ResDenseBlock(xin, growth_rate, n_filter):
    x1 = Conv2D(filters = growth_rate, kernel_size=3,padding='same')(xin)
    x1 = LeakyReLU()(x1)
    x1 = Concatenate()([xin, x1])
    x2 = Conv2D(filters = growth_rate, kernel_size=3,padding='same')(x1)
    x2 = LeakyReLU()(x2)
    x2 = Concatenate()([x1, x2])
    x3 = Conv2D(filters = growth_rate, kernel_size=3,padding='same')(x2)
    x3 = LeakyReLU()(x3)
    x3 = Concatenate()([x2, x3])
    x4 = Conv2D(filters = growth_rate, kernel_size=3,padding='same')(x3)
    x4 = LeakyReLU()(x4)
    x4 = Concatenate()([x3, x4])
    
    x = Conv2D(filters = n_filter, kernel_size=3,padding='same')(x4)
    x = Add()([xin, x])
    # 这里没有使用残差缩放因子
    return x

def createRDBGenerator(blockNum,growth_rate,n_filter):
    '''
    创建生成器
    blockNum：残差稠密块数量
    growth_rate：卷积核数，一般32
    n_filter: 块内最后一层卷积核数，一般64
    '''
    #输入层
    inputLayer = Input(shape=(None,None,3))
    
    #第一层
    firstLayer = Conv2D(filters=growth_rate,kernel_size=3,padding='same')(inputLayer)
    firstLayer = LeakyReLU()(firstLayer)

    #中间层，残差块或RDB    
    middle = firstLayer
    for num in range(blockNum):
        middle = ResDenseBlock(middle,growth_rate,n_filter)
    middle = Conv2D(filters=growth_rate,kernel_size=3,padding='same')(middle)
    middle = Add()([firstLayer,middle])
    
    #还有两个小块
    middle = Conv2D(filters=growth_rate,kernel_size=3,padding='same')(middle)
    middle = LeakyReLU()(middle)
    middle = Conv2D(filters=growth_rate,kernel_size=3,padding='same')(middle)
    middle = LeakyReLU()(middle)
    
    #输出层
    outputLayer = Conv2D(filters=3,kernel_size=3,padding="same",activation='tanh')(middle)
    
    #建模
    model = Model(inputs=inputLayer,outputs=outputLayer)
    
    print('-------------------生成器结构---------------------')
    model.summary()
    print(inputLayer.shape)
    
    return model


def createRDBDiscriminator(lr):
    '''
    创建判别器
    layerNum：中间块数
    filterNum：中间块卷积核数
    lr：学习率
    '''    
    #输入层
    inputLayer = Input(shape=(480,480,3))
    
    #中间层
    middle = Conv2D(64,kernel_size=3,strides=1,padding='same')(inputLayer)
    middle = LeakyReLU()(middle)

    middle = Conv2D(64,kernel_size=3,strides=2,padding='same')(middle)
    middle = LeakyReLU()(middle)
    
    middle = Conv2D(128,kernel_size=3,strides=2,padding='same')(middle)
    middle = LeakyReLU()(middle)
    
    middle = Conv2D(256,kernel_size=3,strides=2,padding='same')(middle)
    middle = LeakyReLU()(middle)
    
    middle = Conv2D(512,kernel_size=3,strides=2,padding='same')(middle)
    middle = LeakyReLU()(middle)
    
    middle = Conv2D(512,kernel_size=3,strides=2,padding='same')(middle)
    middle = LeakyReLU()(middle)
    
    middle = Conv2D(512,kernel_size=3,strides=2,padding='same')(middle)
    middle = LeakyReLU()(middle)
    
    middle = Conv2D(512,kernel_size=3,strides=2,padding='same')(middle)
    middle = LeakyReLU()(middle)
    
    middle = Flatten()(middle)
    middle = Dense(1024,activation=LeakyReLU())(middle)
    
    #输出层
    outputLayer = Dense(1, activation='sigmoid')(middle)
    
    #建模
    model = Model(inputs=inputLayer,outputs=outputLayer)
    #优化器
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    
    print('-------------------判别器结构---------------------')
    model.summary()
    
    return model


# In[16]:


def reverseImg(img):
    '''将图片还原到原数量级'''
    img = (img + 1)*255/2
    return img


def createRDBGan(generator,discriminator,lr):
    '''构建对抗网'''
    discriminator.trainable = False
    #生成器输入
    lowImg = generator.input
    #生成器输出
    fakeHighImg = generator(lowImg)
    #生成器判断
    judge = discriminator(fakeHighImg)
    model = Model(inputs=lowImg,outputs=[judge,fakeHighImg])
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss=['binary_crossentropy', contentLoss],loss_weights=[1e-3, 1], run_eagerly = True)
    
    print('-------------------GAN网络---------------------')
    model.summary()

    return model



