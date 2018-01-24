# -*- coding: utf-8 -*-
# /* Created by PC on 2018/1/24.*/
"""
代码源于以下的地址
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 为两个计算设置随机种子
tf.set_random_seed(1)

#继续使用到手写数字 MNIST 数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

###########设置RNN的参数 #############
# 确定 RNN 的各种参数(hyper-parameters)
lr = 0.001
training_iters = 100000
batch_size = 128
n_inputs = 28   # MNIST data input (img shape: 28*28)
n_steps = 28    # time steps
n_hidden_units = 128   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)

# 定义 x, y 的 placeholder 和 weights, biases 的初始状况
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


#########接着开始定义 RNN 主体结构
def RNN(X, weights, biases):

    ######################################### hidden layer for input to cell
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])


    ########################################### cell
    # 使用 basic LSTM Cell.
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # 对于 lstm 来说, state可被分为(c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)# 初始化全零 state

    #cell的计算，tf.nn.dynamic_rnn使用前需要先确定 inputs 的格式
    '''
    我们要确定 inputs 的格式
    如果 inputs 为 (batches, steps, inputs) ==> time_major=False;
    如果 inputs 为 (steps, batches, inputs) ==> time_major=True;
    这里input为(batches, steps, inputs)
    '''
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    #############################################
    # 最后是 output_layer 和 return 的值
    # #方法一：直接调用final_state 中的 h_state (final_state[1]) 来进行运算 results = tf.matmul(final_state[1], weights['out']) + biases['out']
    #方法二： 调用最后一个 outputs (在这个例子中,和上面的final_state[1]是一样的)
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results

#计算 cost 和 train_op
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)


##########开始训练rnn
#训练时, 不断输出 accuracy, 观看结果
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
        step += 1