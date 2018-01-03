#！/usr/bin/env python
#_*_coding:utf-8_*_
import  tensorflow as tf
import numpy as np

#使用numpy生成100个随机点,这些点作为训练数据
x_data=np.random.rand(100)
#真实的线性模型
y_data=x_data*0.1+0.2

#构造自己的线性模型
#线性模型的参数
b=tf.Variable(0.)
k=tf.Variable(0.)
#构造线性模型
y = k*x_data + b

#二次代价函数:实际y的值与模型总预测的y的值的差求平方，然后取平均值
loss=tf.reduce_mean(tf.square(y_data-y))
#定义一个梯度下降法来进行训练的优化器
optimizer=tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数，我们的目的就是最小化代价函数,目的就是最小化loss4
train=optimizer.minimize(loss)

#初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #进行迭代
    for step in range(201):
        sess.run(train)
        if step%20==0:
            print(step,sess.run([k,b]))
