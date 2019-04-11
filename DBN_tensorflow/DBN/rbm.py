# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class RBM(object):
    def __init__(self,
                 rbm_h_type='bin',
                 rbm_struct=[3072,800],
                 rbm_epochs=10,
                 batch_size=32,
                 cd_k=1,
                 rbm_lr=1e-3):
        self.rbm_h_type=rbm_h_type
        self.n_v = rbm_struct[0]               #可视层784个特征（节点）
        self.n_h = rbm_struct[1]               #隐藏层100
        self.rbm_epochs = rbm_epochs
        self.batch_size = batch_size
        self.cd_k = cd_k
        self.rbm_lr = rbm_lr
        
    ###################
    #    RBM_model    #
    ###################
    
    def build_rbm(self):
        # feed 变量
        self.input_data = tf.placeholder(tf.float32, [None, self.n_v]) # N等于batch_size（训练）或_num_examples（测试）
        # 权值 变量（初始化）
        self.W = tf.Variable(tf.truncated_normal(shape=[self.n_v, self.n_h], stddev=0.1), name='W')
        self.bh = tf.Variable(tf.constant(0.1, shape=[self.n_h]),name='bh')           #隐藏层偏置
        self.bv = tf.Variable(tf.constant(0.1, shape=[self.n_v]),name='bv')            #可视层偏置
        self.parameter = [self.W, self.bh]    #存储参数

        # v0,h0
        v0=self.input_data                 #n*784
        h0,s_h0=self.transform(v0)          #返回向前传播的a值，以及二进制形式的a值 n*100

        # vk,hk   在可视层与隐藏中来回重构迭代
        vk=self.reconstruction(s_h0)        #反向传播，重构可视层
        for k in range(self.cd_k-1):
            _,s_hk=self.transform(vk)
            vk=self.reconstruction(s_hk)
        hk,_=self.transform(vk)

        # upd8
        positive=tf.matmul(tf.expand_dims(v0,-1), tf.expand_dims(h0,1))      #tf.expand_dims使维度在索引的位置加1  [n*784*1]*[n*1*100]
        negative=tf.matmul(tf.expand_dims(vk,-1), tf.expand_dims(hk,1))
        self.w_upd8 = self.W.assign_add(tf.multiply(self.rbm_lr, tf.reduce_mean(tf.subtract(positive, negative), axis=0)))     #更新参数
        self.bh_upd8 = self.bh.assign_add(tf.multiply(self.rbm_lr, tf.reduce_mean(tf.subtract(h0, hk), axis=0)))
        self.bv_upd8 = self.bv.assign_add(tf.multiply(self.rbm_lr, tf.reduce_mean(tf.subtract(v0, vk), 0)))
        self.train_batch_cdk = [self.w_upd8, self.bh_upd8, self.bv_upd8]                #存放参数
        # loss
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(v0, vk))))           #计算损失函数

     #训练rbm
    def train_rbm(self,train_X,sess):
        # 参数设置
        sess.run(tf.global_variables_initializer())
        self.images = train_X
        self._images = train_X
        self._num_examples = train_X.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        # 迭代次数
        for i in range(self.rbm_epochs):
            for _ in range(int(self._num_examples/self.batch_size)): 
                batch = self.next_batch()    #next_batch()返回的是样本
                loss,_= sess.run([self.loss,self.train_batch_cdk],feed_dict={self.input_data: batch})
            print('>>> epoch = {} , loss = {:.4}'.format(i+1,loss))

    #向前传播，返回a，以及二进制形式的a
    def transform(self,v):       #v[n,784]
        v = tf.cast(v, dtype=tf.float32)
        z = tf.add(tf.matmul(v, self.W), self.bh)
        if self.rbm_h_type=='bin':
            prob_h=tf.nn.sigmoid(z)            #激活z后的值 n*100
            state_h= tf.to_float(tf.random_uniform([tf.shape(v)[0],self.n_h])<prob_h)      #转换成0-1二进制 n*100
        else:
            prob_h=z
            state_h=tf.add(prob_h, tf.random_uniform([tf.shape(v)[0],self.n_h]))
        return prob_h,state_h

    #反向传播的激活值
    def reconstruction(self,h):
        z = tf.add(tf.matmul(h, tf.transpose(self.W)), self.bv)
        prob_v=tf.nn.sigmoid(z)
        return prob_v
    
    ###################
    #     RBM_func    #
    ###################
    #批处理
    def next_batch(self,shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
        # Go to the next epoch
        if start + self.batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = self.batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0)
        else:
            self._index_in_epoch += self.batch_size
            end = self._index_in_epoch
            return self._images[start:end]
