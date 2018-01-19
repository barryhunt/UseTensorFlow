# -*- coding: utf-8 -*-
# /* Created by PC on 2018/1/19.*/

'''
Dropout 解决 overfitting,并使用tensorboard呈现效果
'''
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 加载数据
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


########建立 dropout层
#keep_prob是保留概率，要保留的结果所占比例，keep_prob=1的时候，相当于100%保留
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

#添加隐含层和输出层
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

#loss函数（即最优化目标函数）选用交叉熵函数，如果完全相同，交叉熵就等于零
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))  # loss
tf.summary.scalar('loss', cross_entropy)

#train方法（最优化算法）采用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
# 所有的输出
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)


init = tf.global_variables_initializer()
sess.run(init)
for i in range(500):
    # 最后开始train，总共训练500次。
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
    #sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
    if i % 50 == 0:
        # 记录loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)

#cmd切换到F:/Tensorflow文件夹，分别输入tensorboard --logdir=f:/tensorflow/logs

#训练中keep_prob=1时，就可以暴露出overfitting问题。keep_prob=0.5时，dropout就发挥了作用。 我们可以两种参数分别运行程序，对比一下结果。
#当keep_prob=1时，模型对训练数据的适应性优于测试数据，存在overfitting，输出如下： 红线是 train 的误差, 蓝线是 test 的误差

