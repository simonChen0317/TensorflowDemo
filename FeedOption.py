#！/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf
#Feed Feed的作用是在运行时将值传入
#使用placeholder创建占位符
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
output=tf.multiply(input1,input2)
with tf.Session() as sess:
    #feed的数据以字典的形式传入
    print(sess.run(output,feed_dict={input1:[5.0],input2:[3.0]}))
