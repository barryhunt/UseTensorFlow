# -*- coding: utf-8 -*-
# 22 scope (name_scope/variable_scope)
'''
scope 能让你命名变量的时候轻松很多
'''
import tensorflow as tf

'''在 Tensorflow 当中有两种途径生成变量 variable,
一种是 tf.get_variable(),
另一种是 tf.Variable().'''

#在 tf.name_scope() 的框架下分别使用这两种方式
with tf.name_scope("a_name_scope"):
    initializer = tf.constant_initializer(value=1)
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)

    #三个name都一样
    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
    var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(var1.name)        # var1:0 ，使用tf.get_variable()定义的变量不会被tf.name_scope()当中的名字所影响
    print(sess.run(var1))   # [ 1.]

    #虽然 name 都一样, 但是为了不重复变量名, Tensorflow 输出的变量名并不是一样
    print(var2.name)        # a_name_scope/var2:0
    print(sess.run(var2))   # [ 2.]
    print(var21.name)       # a_name_scope/var2_1:0
    print(sess.run(var21))  # [ 2.0999999]
    print(var22.name)       # a_name_scope/var2_2:0
    print(sess.run(var22))  # [ 2.20000005]

'''
可以看出使用 tf.Variable() 定义的时候, 虽然 name 都一样, 但是为了不重复变量名, Tensorflow 输出的变量名并不是一样的. 所以, 本质上 var2, var21, var22 并不是一样的变量. 而另一方面, 使用tf.get_variable()定义的变量不会被tf.name_scope()当中的名字所影响.
'''

############################
#使用 tf.variable_scope(),达到重复利用变量的效果,并搭配 tf.get_variable() 这种方式产生和提取变量
'''
不像 tf.Variable() 每次都会产生新的变量, 
tf.get_variable() 如果遇到了同样名字的变量时, 它会单纯的提取这个同样名字的变量(避免产生新变量). 
而在重复使用的时候, 一定要在代码中强调 scope.reuse_variables(), 
否则系统将会报错, 以为你只是单纯的不小心重复使用到了一个变量
'''
with tf.variable_scope("a_variable_scope") as scope:
    initializer = tf.constant_initializer(value=3)
    var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer) #tf.get_variable
    var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)  #tf.Variable
    var4_reuse = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
    scope.reuse_variables() #一定要在代码中强调
    var3_reuse = tf.get_variable(name='var3',)

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)
    print(var3.name)            # a_variable_scope/var3:0
    print(sess.run(var3))       # [ 3.]
    print(var3_reuse.name)      # a_variable_scope/var3:0
    print(sess.run(var3_reuse)) # [ 3.]

    print(var4.name)            # a_variable_scope/var4:0
    print(sess.run(var4))       # [ 4.]
    print(var4_reuse.name)      # a_variable_scope/var4_1:0
    print(sess.run(var4_reuse)) # [ 4.]

