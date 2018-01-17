#！/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf

with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')

with tf.device('/gpu:1'):
    c = a + b
#通过log_device_placement参数来输出运行每一个运算的设备。
sess = tf.Session(config=tf.ConfigProto(log_device_placement = True))
print(sess .run(c))