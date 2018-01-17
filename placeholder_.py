# -*- coding: utf-8 -*-
# /* Created by PC on 2018/1/16.*/

#placeholder 是 Tensorflow 中的占位符，暂时储存变量
#从外部传入data时，需要用到 tf.placeholder()
import tensorflow as tf

#首先，在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output
ouput = tf.multiply(input1, input2)

#然后，以该形式传输数据：sess.run(***, feed_dict={input: **})
#placeholder 与 feed_dict={} 绑定在一起出现
with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))