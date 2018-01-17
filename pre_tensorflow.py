# -*- coding: utf-8 -*-
# /* Created by PC on 2018/1/16.*/
import tensorflow as tf
import numpy as np

#创建数据
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

#搭建模型
weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))  #用 tf.Variable 来创建描述 y 的参数.
biases=tf.Variable(tf.zeros([1]))

y = weights*x_data + biases

#定义损失函数
loss=tf.reduce_mean(tf.square(y-y_data))

#反向传递误差
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss) #使用Gradient Descent来对loss函数的参数进行更新

#########   训练   ########
##########################
#先初始化所有之前定义的Variable
# init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
init = tf.global_variables_initializer()  # 替换成这样就好

#然后创建会话 Session
sess = tf.Session()
sess.run(init)          # Very important

for step in range(201):
    sess.run(train)
    if step % 20 == 0: #每隔20个输出一次
        print(step, sess.run(weights), sess.run(biases))
#################################