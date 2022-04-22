#!/usr/bin/env python
# coding: utf-8

# In[13]:


from tensorflow.keras.layers import Input,Dense,Conv2D,Flatten,BatchNormalization,UpSampling2D,MaxPooling2D,Activation
from tensorflow.keras.layers import PReLU,Add,Concatenate,LeakyReLU
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



def resBlock(xIn,filterNum):
    '''残差块'''
    x = Conv2D(filters=filterNum,kernel_size=3,padding='same')(xIn)
    x = LeakyReLU()(x)
    x = Conv2D(filters=filterNum,kernel_size=3,padding='same')(x)
    x = Add()([xIn, x])
    return x

def createResGenerator(layerNum,filterNum):
    '''
    创建生成器
    layerNum：残差块数
    filterNum：残差块卷积核数
    '''
    #输入层
    inputLayer = Input(shape=(None,None,3))
    
    #第一层
    firstLayer = Conv2D(filters=filterNum,kernel_size=3,padding='same')(inputLayer)
    firstLayer = LeakyReLU()(firstLayer)

    #中间层，残差块    
    middle = firstLayer
    for num in range(layerNum):
        middle = resBlock(middle,filterNum)   
    middle = Conv2D(filters=filterNum,kernel_size=3,padding='same')(middle)
    middle = Add()([firstLayer,middle])
    
    #还有两个小块
    middle = Conv2D(filters=filterNum,kernel_size=3,padding='same')(middle)
    middle = LeakyReLU()(middle)
    
    middle = Conv2D(filters=filterNum,kernel_size=3,padding='same')(middle)
    middle = LeakyReLU()(middle)
    
    #输出层
    outputLayer = Conv2D(filters=3,kernel_size=3,padding="same",activation='tanh')(middle)
    
    #建模
    model = Model(inputs=inputLayer,outputs=outputLayer)
    
    print('-------------------生成器结构---------------------')
    model.summary()
    print(inputLayer.shape)
    
    return model



def createResDiscriminator(lr):
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


def createResGan(generator,discriminator,lr):
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



