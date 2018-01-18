# -*- coding: utf-8 -*-
# /* Created by PC on 2018/1/18.*/

"""
python 3+. 使用tensorboard可视化训练过程( biase变化过程).
"""

import tensorflow as tf
import numpy as np

#########首先，制作输入源############
##################################
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


##其次，在add_layer中为Weights, biases 绘制变化图表##########
#########################################################
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer #在 add_layer() 方法中添加一个参数 n_layer,用来标识层数
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'): #添加weights的图层
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights) #设置图层中的Weights变化图，使用tf.summary.histogram()方法绘制, 第一个参数是图表的名称,第二个是图表要记录的变量
        with tf.name_scope('biases'): #biases同理
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None: #activation_function可以不绘制变化图
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs




with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

#隐藏层，n_layer=1，第一层神经元
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
#输出层，n_layer=2，表示第一层神经元
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)


############然后，设置loss的变化图##########
#########################################
with tf.name_scope('loss'):#观看loss的变化比较重要. 当loss呈下降的趋势,说明神经网络训练有效果
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    tf.summary.scalar('loss', loss)# loss在tesnorBorad的scalars下面，因为使用的是tf.summary.scalar方法

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

####接着，给所有训练图合并########
##############################
sess = tf.Session()
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("F:/Tensorflow/logs/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)


######最后，训练数据#######
#########################
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged,feed_dict={xs: x_data, ys: y_data})  #merged需要run才能发挥作用的
        writer.add_summary(result, i)   # tf.merge_all_summaries() 方法会对我们所有的 summaries 合并到一起

# 重要！比如文件存放在F:/Tensorflow/logs/文件夹下
# cmd中切换到logs的上级目录，即F:/Tensorflow
#F:\TensorFlow>tensorboard --logdir=f:/tensorflow/logs     可以使用 tensorboard --help 查看tensorboard的详细参数
#Google Chrome中打开shell中提示的地址