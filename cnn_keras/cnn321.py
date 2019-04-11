
'''
fun: 利用xception对数据集专利弄出来矿山20类地物进行分类（不做数据增强），顺便提取softmax分类结果；
time； 2018-6-5
author: tang
'''
import sklearn
import pandas as pd
from keras import models,layers
from keras.callbacks import TensorBoard
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np
from keras.utils import plot_model
#import pydot
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, Flatten, Dropout, SeparableConv2D
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.preprocessing import image
from keras.utils import np_utils
from keras.models import Model,Sequential
#from sklearn.cross_validation import train_test_split
import os

from sklearn.metrics import cohen_kappa_score
from tensorflow import confusion_matrix

import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
# random.seed(68)
from read_image import get_files,get_images

# import keras.backend.tensorflow_backend as KTF
# config = tf.ConfigProto(device_count={'gpu': 0})
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# session = tf.Session(config=config)
# # 设置session
# KTF.set_session(session)

NUM_classes = 20  # 分类的类别个数
def mycnn():
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=(2, 2), strides=(1, 1), padding='same', dim_ordering='tf',
                     input_shape=(15,15,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=256, kernel_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=512, kernel_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=512, kernel_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.3))

    # model.add(Dense(3,activation='softmax))
    model.add(Dense(20, activation='softmax'))
    # model.add(BatchNormalization())
    # model.add(Activation('softmax'))
    model.summary()
    plot_model(model, to_file='cnn.png', show_shapes=True)  # 画出模型结构图
    return model

def run(X_train, y_train,X_test, y_test,train_index,test_index):

    model.fit(X_train, y_train, epochs=1000, batch_size=128,
              callbacks=[TensorBoard(log_dir=r'.\out')])
    loss, score = model.evaluate(X_test, y_test, batch_size=256)  # 得到精度损失
    print('loss为：'+str(loss))
    # print('accuracy为：'+str(score))
    model.save('./out/cnn_sgd_1000.h5' )
    #i = i + 1
    pre = []
    tre = []
    a = model.predict(X_test, batch_size=256)  # +++++++++++++++++++++++++++softmax值
    for i in range(len(a)):
        label = np.argmax(a[i])
        tre1 = np.argmax(y_test[i])  # 真实值标签列表
        pre.append(label)
        tre.append(tre1)
    acc = 0
    for i in range(len(a)):
        if pre[i] == tre[i]:
            acc = acc + 1
    # print(a.shape)
    # print(type(a))
    score = float((acc / len(a)))                      #OA
    kappa = cohen_kappa_score(tre, pre)
    #print(sklearn.metrics.confusion_matrix(tre, pre))
    a = sklearn.metrics.confusion_matrix(tre, pre)  #混淆矩阵

    xsum = np.sum(a, axis=1)
    xsum = np.array(xsum)
    ysum = np.sum(a, axis=0)
    ysum = np.array(ysum)
    # print(xsum)                                   # 举证右求和
    # print(ysum)
    recall = []
    precision = []
    f1_measure = []
    for i in range(len(xsum)):
        reca = (a[i][i] / xsum[i])
        recall.append(reca)

        prec = (a[i][i] / ysum[i])
        precision.append(prec)

        f1 = 2 * recall[i] * precision[i] / (recall[i] + precision[i])
        f1_measure.append(f1)
    print(a)                                             #混淆矩阵
    print('recall' + str(recall))                       # 召回率
    print("precisoin" + str(precision))                 # 准确率
    print("f1_measure" + str(f1_measure))              # f1measure
    print("f1_score" + str(np.average(f1_measure)))   # f1分数
    print("kappa"+str(kappa))                         #kappa系数
    print("OA"+str(score))                         #kappa系数
    return


if __name__ == '__main__':
    # base_model = VGG16(include_top=False, weights='imagenet',input_shape=(128,128,3))
    model = mycnn()

    from keras.optimizers import SGD#SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer = 'sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    train_dir = r'D:\tang\桌面数据\暑期DBN\data4\cnn_test\train_merge'  # ++++++++++++++++++++++++++++++
    img_path_list, y_train,train_index = get_files(train_dir)  # 获得y-标签列表
    X_train = get_images(img_path_list,target_size=15)  # 获得x-多维数组（预处理后的）
    y_train = np_utils.to_categorical(y_train, NUM_classes)  # 转换成one-hot编码

    test_dir = r'D:\tang\桌面数据\暑期DBN\data4\cnn_test\test_merge'  # ++++++++++++++++++++++++++++++
    img_path_list1, y_test, test_index = get_files(test_dir)  # 获得y-标签列表
    X_test = get_images(img_path_list1,target_size=15)  # 获得x-多维数组（预处理后的）
    y_test = np_utils.to_categorical(y_test, NUM_classes)  # 转换成one-hot编码
    run(X_train, y_train, X_test, y_test,train_index,test_index)

# img_path = 'plane.jpg'
# img = image.load_img(img_path, target_size=(299, 299))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# preds = model.predict(x)
# print('Predicted:', decode_predictions(preds, top=3)[0])