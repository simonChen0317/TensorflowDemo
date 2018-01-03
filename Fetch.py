#！/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf
#Fetch 同时运行多个计算
input1=tf.constant(5.0)
input2=tf.constant(2.0)
input3=tf.constant(3.0)

add=tf.add(input2,input3)
mul= tf.multiply(input1,add)
with tf.Session() as sess:
    result=sess.run([add,mul])
    print(result)

