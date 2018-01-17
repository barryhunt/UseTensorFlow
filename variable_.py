# -*- coding: utf-8 -*-
# /* Created by PC on 2018/1/16.*/
import tensorflow as tf
#定义变量
state=tf.Variable(0,name='counter')

# 定义常量 one
one = tf.constant(1)

# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, one)

# 将 State 更新成 new_value
update = tf.assign(state, new_value)

######### 如果定义 Variable, 就一定要 initialize#########
#####################################
# init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
init = tf.global_variables_initializer()  # 替换成这样就好

# 使用 Session
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))   #直接 print(state) 不起作用