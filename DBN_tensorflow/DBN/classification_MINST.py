# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(r'E:\py3\test_myDBN')
import numpy as np
import tensorflow as tf
from my_model.dbn import DBN
np.random.seed(1337)  # for reproducibility
from read_data.myinput import get_files,get_images
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#from a.mode.dbn import DBN


image_train = r'C:\Users\Administrator\Desktop\data\train'
imgdir_list,Y_train = get_files(image_train)
X_train = get_images(imgdir_list)
a = np.random.rand(len(Y_train), 2)
for p in range(len(Y_train)):
    if Y_train[p] == 0:
        a[p][0] = 1
        a[p][1] = 0

    elif Y_train[p] == 1:
        a[p][0] = 0
        a[p][1] = 1
Y_train = a
X_train = X_train/256
print(Y_train[:20])
print(X_train.shape)        #输入模型的数据
print(Y_train.shape)

image_test = r'C:\Users\Administrator\Desktop\data\test'
imgdir_list,Y_test = get_files(image_test)
X_test = get_images(imgdir_list)  #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
b = np.random.rand(len(Y_test),2)
for p in range(len(Y_test)):
    if Y_test[p] == 0:
        b[p][0] = 1
        b[p][1] = 0

    elif Y_test[p] == 1:
        b[p][0] = 0
        b[p][1] = 1
X_test = X_test/256
Y_test = b           #yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
print(type(Y_test[0]))

# Splitting data
#X_train, X_test, Y_train, Y_test = mnist.train.images,mnist.test.images ,mnist.train.labels, mnist.test.labels


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training
classifier = DBN(output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_fuc='cross_entropy',
                 use_for='classification',
                 dbn_lr=1e-3,
                 dbn_epochs=100,
                 dbn_struct=[3072, 800, 200,2],
                 rbm_h_type='bin',
                 rbm_epochs=10,
                 batch_size=32,
                 cd_k=1,
                 rbm_lr=1e-3)

classifier.build_dbn()
classifier.train_dbn(X_train, Y_train,sess)

# Test
Y_pred = classifier.test_dbn(X_test, Y_test,sess)
print(Y_pred[:10])
#import matplotlib.pyplot as plt
#PX=range(0,len(Y_pred))
#plt.figure(1)  # 选择图表1
#plt.plot(PX, Y_test,'r')
#plt.plot(PX, Y_pred,'b')
#plt.show()