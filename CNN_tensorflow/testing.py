#coding=utf-8

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from CNNt_multipy import input_data
import numpy as np
from CNNt_multipy import model
import os
from sklearn.metrics import confusion_matrix


#从指定目录中选取图片
def get_images(img_dir):
    #return image in the type of array
    array = np.random.rand(300,128,128,3)
    n =len(img_dir)
    #ind = np.random.randint(0,n)       #0-n之间随机取一个整数
    # img_dir = os.path.join(train,files[ind])
    for i in range(n):
        image = Image.open(img_dir[i])
        image = image.resize([128, 128])
        img_arr =  np.array(image)
        array[i] = img_arr

    return array

#a = get_one_image(r'C:\Users\Administrator\Desktop\data\test')

def evaluate_images():
    #测试集路径
    image_test = r'C:\Users\Administrator\Desktop\data\test'
    imgdir_list,label_list = input_data.get_files(image_test)
    # print (imgdir_list)
    # print (label_list)

    image_array = get_images(imgdir_list)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 3
        num = len(label_list)

        #转换图片格式
        image = tf.cast(image_array,tf.float32)


        # 图片原来是三维的 [208, 208, 3] 重新定义图片形状 改为一个4D  四维的 tensor
        image = tf.reshape(image, [num, 128, 128, 3])
        #print (image)
        print(image.shape)
        print('**********************')
        logit = model.inference(image, num, N_CLASSES)
        # 因为 inference 的返回没有用激活函数，所以在这里对结果用softmax 激活
        logit = tf.nn.softmax(logit)

        #用最原始的输入数据方式向模型输入数据placeholder
        #x = tf.placeholder(tf.float32,shape=[200,128,128,3])

        # 我们存放模型的路径
        logs_train_dir = r'C:\Users\Administrator\Desktop\data\saveNet'
        #定义saver
        saver = tf.train.Saver()

        result = []
        with tf.Session() as sess:
            print("从指定的模型中加载数据。。")
            #将模型加载到sess中
            # tf.get_variable_scope().reuse_variables()

            #验证之前是否已经保存了检查点文件
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('\\')[-1].split('-')[-1]
                # 使用saver.restore()方法，重载模型的参数，用于训练
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('模型加载成功, 训练的步数为 %s' % global_step)
            else:
                print('模型加载失败，，，文件没有找到')
                # 将图片输入到模型计算
            prediction = sess.run(logit)

            # 获取输出结果中最大概率的索引
            print (prediction)           #num*NUM_CLASS
            print (prediction.shape)

            for i in range(num):
                p = np.argmax(prediction[i])
                result.append(int(p))
            # print (result)
            kk=0
            for i in range(num):
                if (result[i] == label_list[i]):
                    kk = kk + 1

            print('acc rate is %f' % float(kk / num))
            print('###########################################')



    return label_list,result

def my_confusion_matrix(y_true, y_pred):
    #输出混淆矩阵
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels = labels)
    print ("confusion_matrix(left labels: y_true, up labels: y_pred):")
    print ("labels"," ",end='')
    for i in range(len(labels)):
        print (labels[i]," ",end='')
    print('\n')
    for i in range(len(conf_mat)):
        print (i," ",end='')
        for j in range(len(conf_mat[i])):
            print (conf_mat[i][j]," ",end='')
        print('\n')
    print

# 测试
y_true, y_pred = evaluate_images()
my_confusion_matrix(y_true, y_pred)




