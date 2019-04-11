# -*- coding: utf-8 -*-
import tensorflow as tf

from my_model.rbm import RBM

class RBMs(object):
    def __init__(self,
                 rbm_h_type='bin',
                 rbms_struct=[3072, 800, 200],
                 rbm_epochs=10,
                 batch_size=32,
                 cd_k=1,
                 rbm_lr=1e-3):
        self.rbm_h_type=rbm_h_type
        self.rbms_struct = rbms_struct
        self.rbm_epochs = rbm_epochs
        self.batch_size = batch_size
        self.cd_k = cd_k
        self.rbm_lr = rbm_lr
        
    ####################
    #    RBMS_model    #
    ####################
    #构建rbms
    def build_rbms(self):
        # feed 变量
        self.input_data = tf.placeholder(tf.float32, [None, self.rbms_struct[0]]) # N等于_num_examples或batch_size  n*784
        # 构建rmbs
        self.rbm_list = list()
        for i in range(len(self.rbms_struct) -1):
            n_v = self.rbms_struct[i]                #可视层
            n_h = self.rbms_struct[i+1]              #隐藏层
            rbm = RBM(rbm_h_type=self.rbm_h_type,
                 rbm_struct=[n_v,n_h],
                 rbm_epochs=self.rbm_epochs,
                 batch_size=self.batch_size,
                 cd_k=self.cd_k,
                 rbm_lr=self.rbm_lr)
            rbm.build_rbm()
            self.rbm_list.append(rbm) # 将rbm0  rbm1 加入list
    #训练rbms
    def train_rbms(self,train_X,sess):
        next_data = train_X # 这个next_data是实数
        for i,rbm in enumerate(self.rbm_list):          #enumerate返回下标和元素
            print('>>> Training RBM-{}:'.format(i+1))
            # 训练第i个RBM（按batch），train函数迭代训练所有批次，无返回值
            rbm.train_rbm(next_data,sess)
            # 得到transform值（train_X）即向前传播的a值---相当于下一个rbm的输入层（可视层）
            next_data,_ = sess.run(rbm.transform(next_data))
    
#    def transform(self,data_x):
#        next_data = data_x # 这个next_data是tf变量
#        for rbm in self.rbm_list:
#            next_data = rbm.transform(next_data)
#        return next_data