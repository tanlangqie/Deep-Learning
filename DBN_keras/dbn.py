'''
fun:用DBN寻找最优参数组合，数据为训练集验证集
time:2018-7-12
'''
import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import classification_report
from sklearn.neural_network import BernoulliRBM
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils
import pandas as pd
import os
import json
import pickle
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 1})))
from read_train import readpoint


class DBN():
    def __init__(
            self,
            train_x, train_y,
            test_x, test_y,
            layers,                 #rbm层数【200，400,300】
            outputs,                #输出
            rbm_iters,  #rbm迭代次数
            rbm_lr,               #学习率，
            epochs=25,                 #dbn迭代次数
            fine_tune_batch_size=2048,
            rbm_dir=None,
            outdir="tmp/",
            logdir="logs/"

    ):
        self.pre = [],
        self.tre = [],
        self.hidden_sizes = layers              #rbm层数
        self.outputs = outputs

        self.train_x = train_x                  #训练集
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        self.pre = []
        self.tre = []
        self.rbm_learning_rate = rbm_lr             #学习率
        self.rbm_iters = rbm_iters                  #rbm迭代次数

        self.epochs = epochs
        self.nn_batch_size = fine_tune_batch_size        #微调批次

        self.rbm_weights = []
        self.rbm_biases = []
        self.rbm_h_act = []

        self.model = None
        self.history = None

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        if outdir[-1] != '/':
            outdir = outdir + '/'

        self.outdir = outdir
        self.logdir = logdir


    def my_confusion_matrix(self,y_true, y_pred):
        # 输出混淆矩阵
        labels = list(set(y_true))
        conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
        print("confusion_matrix(left labels: y_true, up labels: y_pred):")
        print("labels", " ", end='')
        for i in range(len(labels)):
            print(labels[i], " ", end='')
        print('\n')
        for i in range(len(conf_mat)):
            print(i, " ", end='')
            for j in range(len(conf_mat[i])):
                print(conf_mat[i][j], " ", end='')
            print('\n')
        print('\n')
        # print('混淆矩阵-----')
        # print(type(conf_mat))
        # print(conf_mat)
        print('kappa系数--------------')
        # 输出kappa系数
        dataMat = np.array(conf_mat)
        # print(dataMat)
        P0 = 0.0
        for i in range(len(dataMat)):
            P0 += dataMat[i][i] * 1.0
        xsum = np.sum(dataMat, axis=1)
        xsum = np.array(xsum)
        ysum = np.sum(dataMat, axis=0)
        ysum = np.array(ysum)
        t = 0
        for i in range(len(xsum)):
            t += (xsum[i] * ysum[i]) * 1.0
        # xsum是个k行1列的向量，ysum是个1行k列的向量
        Pe = float(t / (dataMat.sum() ** 2))
        P0 = float(P0 / dataMat.sum() * 1.0)
        cohens_coefficient = float((P0 - Pe) / (1 - Pe))
        print('DBN的kappa系数为: %f' % cohens_coefficient)
        return cohens_coefficient

    #预训练
    def pretrain(self, save=True):

        visual_layer = self.train_x            #训练集

        for i in range(len(self.hidden_sizes)):
            print("[DBN] Layer {} Pre-Training".format(i + 1))

            rbm = BernoulliRBM(n_components=self.hidden_sizes[i], n_iter=self.rbm_iters,
                               learning_rate=self.rbm_learning_rate, random_state=16,verbose=0, batch_size=2048)
            rbm.fit(visual_layer)                #训练
            self.rbm_weights.append(rbm.components_)       #权重矩阵
            self.rbm_biases.append(rbm.intercept_hidden_)
            self.rbm_h_act.append(rbm.transform(visual_layer))

            visual_layer = self.rbm_h_act[-1]


    #构建dbn结构
    def finetune(self):
        model = Sequential()
        for i in range(len(self.hidden_sizes)):               #对每一个rbm
                                                              #hidden_sizes =[400,200,60,10]
            if i == 0:
                model.add(Dense(self.hidden_sizes[i], activation='relu', input_dim=self.train_x.shape[1],
                                name='rbm_{}'.format(i)))
            else:
                model.add(Dense(self.hidden_sizes[i], activation='relu', name='rbm_{}'.format(i)))

        model.add(Dense(self.outputs, activation='softmax'))           #
        model.compile(optimizer='SGD',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        for i in range(len(self.hidden_sizes)):
            layer = model.get_layer('rbm_{}'.format(i))
            layer.set_weights([self.rbm_weights[i].transpose(), self.rbm_biases[i]])

        #checkpointer = ModelCheckpoint(filepath=self.outdir + "dbn_weights.hdf5", verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=self.logdir)

        self.history = model.fit(trainx, trainy,
                                     epochs=self.epochs,
                                     batch_size=self.nn_batch_size,verbose=1,#validation_split=0.2,
                                     callbacks=[ tensorboard])
        self.model = model

        a = model.predict(self.test_x, batch_size=256)  # +++++++++++++++++++++++++++
        acc = 0.0
        for i in range(len(a)):
            label = int(np.argmax(a[i]))
            label2 = int(np.argmax(self.test_y[i]))
            self.pre.append(label)
            self.tre.append(label2)
            if label== label2:
                acc = acc + 1

        score = float((acc / len(a)))
        print('DBN的accuracy为：%f' % score)
        kappa_ = self.my_confusion_matrix(self.tre,self.pre)
        return kappa_,score



if __name__ == '__main__':
    num_class = 20          #数据有20个类别
    fs = 106                #特征数量
    trainx, trainy,_p = readpoint(r'C:\Users\1\Desktop\data4\train1.txt',num_class,fs)

    testx,testy,_= readpoint(r'C:\Users\1\Desktop\data4\val1.txt', num_class,fs)

    print('trainx的shape为——{}'.format(trainx.shape))
    print('trainy的shape为——{}'.format(trainy.shape))
    print('testx的shape为——{}'.format(testx.shape))
    print('testy的shape为——{}'.format(testy.shape))

    cengshu = [3]
    jiedian = [250]
    # cengshu = [1]
    # jiedian = [60]
    suiji = 10
    for i in cengshu:
        for j in jiedian:
            print('这次训练有--------{}层,其中每层有--------{}个节点'.format(i, j))
            mylist = []
            mylist.append(j)
            mylayer = mylist * i
            # print(mylayer)
            suiji += 31
            np.random.seed(suiji)
            dbn = DBN(train_x=trainx, train_y=trainy,
                      test_x=testx, test_y=testy,
                      layers=mylayer,
                      outputs=20,
                      rbm_iters=10,
                      rbm_lr=0.0001,
                      epochs=500,
                      fine_tune_batch_size=2048,
                      outdir=r"D:\tang\jiangxia\code4\output",
                      logdir=r"D:\tang\jiangxia\code4\output\log",

                      )
            dbn.pretrain(save=True)
            kappa_, acc = dbn.finetune()
