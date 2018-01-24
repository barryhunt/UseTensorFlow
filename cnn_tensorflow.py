# -*- coding: utf-8 -*-
# /* Created by PC on 2018/1/22.*/

'''
比较流行的一种搭建结构是这样, 从下到上的顺序, 首先是输入的图片(image), 经过一层卷积层 (convolution), 然后在用池化(pooling)方式处理卷积的信息, 这里使用的是 max pooling 的方式. 然后在经过一次同样的处理, 把得到的第二次处理的信息传入两层全连接的神经层 (fully connected),这也是一般的两层神经网络层,最后在接上一个分类器(classifier)进行分类预测.
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
'''
train-images-idx3-ubyte.gz	训练集图片 - 55000 张 训练图片, 5000 张 验证图片
train-labels-idx1-ubyte.gz	训练集图片对应的数字标签
t10k-images-idx3-ubyte.gz	测试集图片 - 10000 张 图片
t10k-labels-idx1-ubyte.gz	测试集图片对应的数字标签
'''
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape): #定义Weight变量
    initial = tf.truncated_normal(shape, stddev=0.1)  #产生随机变量来进行初始化
    return tf.Variable(initial)  #返回变量的参数

def bias_variable(shape): #同理
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):  #定义卷积(padding)
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #tf.nn.conv2d函数是tensoflow里面的二维的卷积函数，x是图片的所有参数，W是此卷积层的权重，步长strides=[1,1,1,1]
    #strides[0]和strides[3]的两个1是默认值，stride [1,1,1,1]中间两个1代表padding时在x方向运动一步，y方向运动一步，padding时我们选的是一次一步，padding采用的方式是SAME

def max_pool_2x2(x): #定义池化层(pooling)
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #padding时我们选的是一次一步，也就是strides[1]=strides[2]=1，这样得到的图片尺寸没有变化.我们采用pooling来稀疏化参数
    #



###### 图片处理 #######
######################
#  定义placeholder，输入数据
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)  #同时，定义dropout的placeholder，它是解决过拟合的有效手段
x_image = tf.reshape(xs, [-1, 28, 28, 1])
#把xs的形状变成[-1,28,28,1]，-1代表先不考虑输入的图片例子多少这个维度，
# 后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3


## 第一个卷积层convolution ##
W_conv1 = weight_variable([5,5, 1,32]) #先定义本层的Weight，本层我们的卷积核patch的大小是5x5，因为黑白图片channel是1所以输入是1，输出是32个featuremap
b_conv1 = bias_variable([32])     #定义bias，它的大小是32个长度
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#定义卷积神经网络的第一个卷积层h_conv1=conv2d(x_image,W_conv1)+b_conv1

#然后在用池化(pooling)方式处理卷积的信息, 这里使用的是 max pooling 的方式
#tf.nn.relu（修正线性单元）对h_conv1进行非线性处理，也就是激活函数来处理，采用了SAME的padding方式，输出图片的大小没有变化依然是28x28，只是厚度变为了32
h_pool1 = max_pool_2x2(h_conv1)#然后经过pooling的处理，输出大小就变为了14x14x32

## 第二个卷积层 ##
#本层我们的输入就是上一层的输出，本层我们的卷积核patch的大小是5x5，有32个featuremap所以输入就是32，输出定为64
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 输出大小 14x14x64

#池化(pooling)
h_pool2 = max_pool_2x2(h_conv2)                                         # 输出大小 7x7x64


## 第一个全连接层 ##
W_fc1 = weight_variable([7*7*64, 1024])#weight_variable的shape输入就是第二个卷积层展平了的输出大小: 7x7x64
b_fc1 = bias_variable([1024])  #后面的输出size我们继续扩大，定为1024
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])# 通过tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据, -1表示先不考虑输入图片例子维度, 将上一个输出结果展平
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  #将展平后的h_pool2_flat与本层的W_fc1相乘
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  #考虑过拟合问题，可以加一个dropout的处理

## 第二个全连接层 ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #用softmax分类器（多分类，输出是各个类的概率）,对我们的输出进行分类预测

#--------------------------------------------------------
# 利用交叉熵损失函数来定义我们的cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss

#用tf.train.AdamOptimizer()作为我们的优化器进行优化，使我们的cross_entropy最小
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session() #定义Session
# important
init = tf.global_variables_initializer()
sess.run(init)

#训练数据
for i in range(1000):#假定训练1000步
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})  #注意sess.run()时记得要用feed_dict给众多 placeholder 传入数据
    if i % 50 == 0: #每50步输出一下准确率
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))