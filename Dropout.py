#！/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
#改例子用来模拟过拟合的情况，然后使用Dropout消除过拟合



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
keep_prob =  tf.placeholder(tf.float32)
#学习率，初始值为0.001
lr = tf.Variable(0.001,dtype=tf.float32)

#创建隐藏层1，[784,2000] 2000表示隐藏层的神经元有2000个
W1 = tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
b1 = tf.Variable(tf.zeros([500]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+ b1)
#dropout这个函数可以设置有多少的神经元在工作
L1_drop = tf.nn.dropout(L1,keep_prob)

#创建隐藏层2
W2 = tf.Variable(tf.truncated_normal([500,300],stddev=0.1))
b2 = tf.Variable(tf.zeros([300]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+ b2)
#dropout这个函数可以设置有多少的神经元在工作
L2_drop = tf.nn.dropout(L2,keep_prob)
#创建输出层
W3 = tf.Variable(tf.truncated_normal([300,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop, W3)+b3)

#使用交叉熵代价函数
#loss = tf.reduce_mean(tf.square(y-prediction))
loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#训练，使用AdamOptimizer优化器
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#cast函数转换为32位的float

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(51):
        sess.run(tf.assign(lr,0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})#0.7表示有70%的神经元工作
        learning_rate = sess.run(lr)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("Iter"+ str(epoch) + ",Testing Accuracy " + str(acc)+",Learning_rate= "+ str(learning_rate))