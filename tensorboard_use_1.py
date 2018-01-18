# -*- coding: utf-8 -*-
# /* Created by PC on 2018/1/17.*/


"""
python 3+.如何用TesorBorad可视化整个神经网络结构的过程
"""
import tensorflow as tf

#定义一个添加层的函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):#使用tensorboard多出来的一行，定义大的框架layer
        with tf.name_scope('weights'):#也需要定义每一个’框架‘里面的小部件
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):#同理
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):#同理
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:  #activation_function图层可以暂时忽略。因为当选择用tensorflow中的激励函数（activation function）的时候，tensorflow会默认添加名称
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs


# 为神经网络的输入定义占位符，传输数据
with tf.name_scope('inputs'):#使用tensorboard多出来的一行，with tf.name_scope('inputs')将xs和ys包含进来，形成一个大的图层，参数'inputs'是图层的名字
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 添加隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 添加输出层
prediction = add_layer(l1, 10, 1, activation_function=None)

# 真实值与预测值的误差
with tf.name_scope('loss'): #为loss添加图层
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

with tf.name_scope('train'):#为train添加图层
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

# tf.train.SummaryWriter 将上面‘绘画’出的图保存到一个目录中，以方便后期在浏览器中可以浏览
# 两种版本写法
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
    writer = tf.train.SummaryWriter('F:/Tensorflow/logs/', sess.graph) #第一个参数存放Tensorboard文件的目录，第二个参数需要使用sess.graph
else: # tensorflow version >= 0.12
    writer = tf.summary.FileWriter("F:/Tensorflow/logs/", sess.graph)

# tf.initialize_all_variables()两种版本写法
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# 重要！比如文件存放在F:/Tensorflow/logs/文件夹下
# cmd中切换到logs的上级目录，即F:/Tensorflow
#F:\TensorFlow>tensorboard --logdir=f:/tensorflow/logs     可以使用 tensorboard --help 查看tensorboard的详细参数
#Google Chrome中打开shell中提示的地址