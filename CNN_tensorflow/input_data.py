# coding=utf-8
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 获取文件路径和标签
def get_files(file_dir):
    # file_dir: 文件夹路径
    # return: 乱序后的图片和标签

    forest = []
    label_forest = []
    lake = []
    label_lake = []
    beach = []
    label_beach = []

    # 载入数据路径并写入标签值
    for file in os.listdir(file_dir):
        name = file.split('_')
        if name[0] ==  'forest':
            forest.append(file_dir +'\\'+ file)
            label_forest.append(0)
        elif name[0] ==  'lake':
            lake.append(file_dir +'\\'+ file)
            label_lake.append(1)
        elif name[0] == 'beach':
            beach.append(file_dir +'\\'+ file)
            label_beach.append(2)
    print('There are %d forest\n There are %d lakes\n There are %d beachs'%(len(forest),(len(lake)),(len(beach))))

    #打乱文件顺序
    image_list = np.hstack((forest,lake,beach))                #将两个列表水平顺序连接
    label_list = np.hstack((label_forest,label_lake,label_beach))
    temp = np.array([image_list,label_list,])          #2*1200
    temp = temp.transpose()    #转置      1200*2
    np.random.shuffle(temp)     #随机打乱顺序

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    print(label_list[:3])
    print(type(label_list[0]))
    label_list = [int(i) for i in label_list]             #将字符型标签转化成整型
    # print (image_list[:5])
    # print (label_list[:5])
    return image_list, label_list



#生成相同大小的批次
def get_batch(image,label,image_W, image_H, batch_size,capacity):
    # image,label  要生成batch的图像和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量
    # return: 图像和标签的batch

    #将python。list类型转换成tf能够识别的格式
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])    #生成一个 图片路径 和 标签 的队列（同一个队列）
    image_contents = tf.read_file(input_queue[0])    #读图片
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)

    #统一图片大小
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,   # 线程
                                              capacity=capacity)
    return image_batch, label_batch



# import matplotlib.pyplot as plt
#
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 64
# IMG_H = 64
#
# image_list, label_list = get_files(file_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#     i = 0
#     coord = tf.train.Coordinator()       #创建一个协调器，管理线程 
#     threads = tf.train.start_queue_runners(coord=coord)      #启动QueueRunner, 此时数据队列已经进队。
#     try:
#         while not coord.should_stop() and i < 5:        # coord.should_stop() 返回 true 时也就是 数据读完了应该调用 coord.request_stop()
#             img, label = sess.run([image_batch, label_batch])
#
#             for j in np.arange(BATCH_SIZE):
#                 print("label: %d" % label[j])
#                 plt.imshow(img[j, :, :, :])
#                 plt.show()
#             i += 1
    #队列中没有数据
#     except tf.errors.OutOfRangeError:
#         print("done!")
#     finally:
#         coord.request_stop()
#     coord.join(threads)