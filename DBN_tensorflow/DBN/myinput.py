# coding=utf-8
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 以列表形式返回图片的路径和标签
def get_files(file_dir):
    # file_dir: 文件夹路径
    # return: 乱序后的图片和标签

    forest = []
    label_forest = []
    lake = []
    label_lake = []


    # 载入数据路径并写入标签值
    for file in os.listdir(file_dir):
        name = file.split('_')
        if name[0] ==  'forest':
            forest.append(file_dir +'\\'+ file)
            label_forest.append(0)
        elif name[0] ==  'lake':
            lake.append(file_dir +'\\'+ file)
            label_lake.append(1)

    print('There are %d forest\n There are %d lakes\n '%(len(forest),(len(lake))))

    #打乱文件顺序
    image_list = np.hstack((forest,lake))                #将两个列表水平顺序连接
    label_list = np.hstack((label_forest,label_lake))
    temp = np.array([image_list,label_list,])          #2*1200
    temp = temp.transpose()    #转置      1200*2
    np.random.shuffle(temp)     #随机打乱顺序

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    #print(label_list[:3])
    #print(type(label_list[0]))
    label_list = [int(i) for i in label_list]             #将字符型标签转化成整型
    # print (image_list[:5])
    # print (label_list[:5])
    return image_list, label_list

#从指定目录中选取图片
def get_images(img_dir):
    #return image in the type of array
    n = len(img_dir)

    array = np.random.rand(n, 32, 32, 3)
    #array = tf.Variable(tf.truncated_normal(shape=[n, 128, 128, 3], stddev=0.1, dtype=tf.float32), name='img_arr')
    # ind = np.random.randint(0,n)       #0-n之间随机取一个整数
    # img_dir = os.path.join(train,files[ind])

    for i in range(n):
        image = Image.open(img_dir[i])
        # 统一图片大小
        image = image.resize([32, 32])
        img_arr = np.array(image)
        array[i] = img_arr

    array = array.reshape(n,-1)
    #array = tf.cast(array, tf.float32)
    return array


