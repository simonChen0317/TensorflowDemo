#！/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf
import numpy as np
#python 画图包
import matplotlib.pyplot as plt
#使用numpy生成200个随机点，最终生成200*1的矩阵
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
#生成干扰项
noise = np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data) + noise

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#构建一个简单的神经网络来实现回归计算
#定义神经网络中间层,这里的1表示输入层为一个神经元，10表示中间层时10个神经元
#初始化中间层的权值
Weight_L1=tf.Variable(tf.random_normal([1,10]))
#初始化中间层的偏置值
biases_L1=tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weight_L1) + biases_L1
#中间层的输出，用正切函数作为激活函数
L1=tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络的输出层
Weight_L2=tf.Variable(tf.random_normal([10,1]))
biases_L2=tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weight_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

#二次代价函数
loss=tf.reduce_mean(tf.square(y - prediction))
#使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(4000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    #获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()
