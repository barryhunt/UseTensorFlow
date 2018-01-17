# -*- coding: utf-8 -*-
# /* Created by PC on 2018/1/16.*/
import tensorflow as tf

# create two matrixes
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1,matrix2)  #两个矩阵相乘

####    两种形式使用会话控制 Session   ##########
# product 不是直接计算的步骤，使用 Session 来激活 product 并得到计算结果
#  method 1
sess = tf.Session()
result1 = sess.run(product)
print(result1)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)




