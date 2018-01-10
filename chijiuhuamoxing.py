#！/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf

#声明两个变量并计算它们的和。
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
result = v1 + v2
#声明tf.train.Saver类用于保存模型。
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    #将模型保存到Save_Model/model.ckpt文件，Save_Model文件夹必须事先建好。在生成的文件中model.ckpt.meta保存了Tensorflow计算图的结构。
    #model.ckpt保存了Tensorflow程序中每一个变量的取值。checkpoint文件保存了一个目录下所有的模型文件列表。
    saver.save(sess, "Save_Model/model.ckpt")