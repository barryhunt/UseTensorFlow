# -*- coding: utf-8 -*-
# /* Created by PC on 2018/1/17.*/
'''
主要分成几个步骤：
1、定义一个添加层的函数def add_layer
2、导入数据
3、使用def add_layer函数开始定义神经层
4、开始训练
'''
import tensorflow as tf
import numpy as np

####################定义一个添加层的函数可以很容易的添加神经层,为之后的添加省下不少时间.
#首先，定义添加神经层的函数def add_layer() 有四个参数：输入值、输入的大小、输出的大小和激励函数，我们设定默认的激励函数是None
def add_layer(inputs, in_size, out_size, activation_function=None):
    #初始化weights和biases，随机变量(normal distribution)会比全部为0要好很多
    Weights = tf.Variable(tf.random_normal([in_size, out_size])) #in_size, out_size分别为行、列
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  #biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1

    #定义Wx_plus_b, 即神经网络未激活的值，tf.matmul()是矩阵的乘法
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    #当activation_function——激励函数为None时，输出就是当前的预测值——Wx_plus_b，不为None时，就把Wx_plus_b传到activation_function()函数中得到输出
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

############导入数据,
#加了一个noise,这样看起来会更像真实情况
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

#利用占位符定义我们所需的神经网络的输入,tf.placeholder()就是代表占位符,None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


############开始定义神经层，神经层都包括输入层、隐藏层和输出层
#输入层和输出层结构一样，比如，输入层有1个属性， 所以我们就有1个输入，1个输出，隐藏层我们可以自己假设，这里为10
#开始定义隐藏层，利用之前的add_layer()函数，这里使用 Tensorflow 自带的激励函数tf.nn.relu
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

#接着，定义输出层。此时的输入就是隐藏层的输出——l1，输入有10层（隐藏层的输出层），输出有1层。
prediction = add_layer(l1, 10, 1, activation_function=None)

#然后，计算预测值prediction和真实值的误差，对二者差的平方求和再取平均
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

#关键的一步，提升它的准确率，这里取的是0.1，代表以0.1的收缩率来最小化误差loss，通常都小于1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#使用变量，先进行初始化，这是必不可少的
# init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
init = tf.global_variables_initializer()  # 替换成这样就好

#定义Session，并用 Session 来执行 init 初始化步骤，只有session.run()才会执行我们定义的运算
sess = tf.Session()
sess.run(init)

######训练开始
for i in range(1000):
    # training，train_step是学习的内容
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0: #每50个输出一下机器学习的误差
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))