#！/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf
#tensorflow中有一下优化器：
# tf.train.GradientDescentOptimizer
# tf.train.AdadeltaOptimizer
# tf.train.AdagradOptimizer
# tf.train.AdagradDAOptimizer
# tf.train.MomentumOptimizer
# tf.train.AdamOptimizer
# tf.train.FtrlOptimizer
# tf.train.ProximalGradientDescentOptimizer
# tf.train.ProximalAdagradOptimizer
# tf.train.RMSPropOptimizer
#各种优化器对比：
#标准梯度下降法： 先计算所有样本汇总误差，然后根据总误差来更新权值
#随机梯度下降法： 随机抽取一个样本来计算误差，然后更新权值
#批量梯度下降法： 这是一种折中的方案，从总样本中选取一个批次（比如10000个样本，随机抽取100个样本作为一个batch），然后计算这个batch的总误差，根据总误差来更新权值


#！/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
#载入数据集,MNIST_data是数据集的路径，此时是改工程下的MNIST_data文件夹下。one_hot=True将标签转换为只有0和1的形式，某一位是1其他位都是0
#下边这条语句会从网站上下载手写字体的数据集，可能会下载不下来，最好是下下载下来，然后放到MNIST_data下。
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#定义每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeHolder。这里的[None,784]表示100行784列。这里是将每个28*28像素的图片变成一个1*784的一维数组对待。
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

#创建一个简单的神经网络,改神经网络没有隐藏层
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W)+b)

#使用交叉熵
#loss = tf.reduce_mean(tf.square(y-prediction))
loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#使用其他的优化器
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#cast函数转换为32位的float

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc =  sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter"+ str(epoch) + ",Testing Accuracy " + str(acc))
