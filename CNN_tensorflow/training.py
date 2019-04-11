# coding=utf-8
import os
import numpy as np
import tensorflow as tf
from CNNt_multipy import input_data
from CNNt_multipy import model

N_CLASSES = 3  # 2个输出神经元，［1，0］ 或者 ［0，1］猫和狗的概率
IMG_W = 128  # 重新定义图片的大小，图片如果过大则训练比较慢
IMG_H = 128
BATCH_SIZE = 32  #每批数据的大小
CAPACITY = 256
MAX_STEP = 5000 # 训练的步数，应当 >= 10000
learning_rate = 0.0001 # 学习率，建议刚开始的 learning_rate <= 0.0001

def run_training():

    # 数据集
    train_dir = r'C:\Users\Administrator\Desktop\data\train'
    #logs_train_dir 存放训练模型的过程的数据，在tensorboard 中查看
    logs_train_dir = r'C:\Users\Administrator\Desktop\data\saveNet'

    # 获取图片和标签集
    train, train_label = input_data.get_files(train_dir)
    # 生成批次
    train_batch, train_label_batch = input_data.get_batch(train,train_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)

    # 进入模型
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    # 获取 loss
    train_loss = model.losses(train_logits, train_label_batch)
    # 训练
    train_op = model.trainning(train_loss, learning_rate)
    # 获取准确率
    train__acc = model.evaluation(train_logits, train_label_batch)

    # 所有的即时数据都要在图表构建阶段合并至一个操作（op）中。
    summary_op = tf.summary.merge_all()
    sess = tf.Session()

    # 实例化一个tf.train.SummaryWriter，用于写入包含了图表本身和即时数据具体值的事件文件。
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

    # 为了得到可以用来后续恢复模型以进一步训练或评估的检查点文件（checkpoint file），我们实例化一个tf.train.Saver。
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                #每次运行summary_op时，都会往事件文件中写入最新的即时数据，
                # 函数的输出会传入事件文件读写器（writer）的add_summary()函数
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or (step + 1) == MAX_STEP:
                #在训练循环中，将定期调用saver.save()方法，向训练文件夹中写入包含了当前所有可训练变量值得检查点文件。
                # 每隔1000步保存一下模型，模型保存在 checkpoint_path 中
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

run_training()