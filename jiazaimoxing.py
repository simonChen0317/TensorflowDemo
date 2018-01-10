#！/usr/bin/env python
#_*_coding:utf-8_*_
#加载模型，在加载模型时也需要先定义TensorFlow计算图上的所有计算，并声明了一个tf.train.Saver()类。
# 与持久化模型不同之处在于没有运行变量的初始化过程
import tensorflow as tf

# #使用和保存模型代码中一样的方式来声明变量。
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
# result = v1 + v2
# #声明tf.train.Saver类用于保存模型。
# saver = tf.train.Saver()

saver = tf.train.import_meta_graph("Save_Model/model.ckpt.meta")

with tf.Session() as sess:
    #加载以保存的模型，并通过以保存的模型中变量的值来计算加法。
    saver.restore(sess,"Save_Model/model.ckpt")
    #通过张量的名称来获取张量。
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
    # print(sess.run(result))