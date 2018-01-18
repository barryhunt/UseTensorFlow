# -*- coding: utf-8 -*-
# /* Created by PC on 2018/1/17.*/
'''
主要包括以下几种训练优化方法:

0、tf.train.GradientDescentOptimizer

1、 随机梯度下降
是梯度下降的batch版本，我们首先将其分成n个batch，每个batch包含m个样本。我们每次更新都利用一个batch的数据，而非整个训练集。
优点：（1）、当训练数据太多时，利用整个数据集更新往往时间上不显示。batch的方法可以减少机器的压力，并且可以更快地收敛。
（2）、当训练集有很多冗余时（类似的样本出现多次），batch方法收敛更快。以一个极端情况为例，若训练集前一半和后一半梯度相同。那么如果前一半作为一个batch，后一半作为另一个batch，那么在一次遍历训练集时，batch的方法向最优解前进两个step，而整体的方法只前进一个step
缺点：其更新方向完全依赖于当前的batch，因而其更新十分不稳定，解决这一问题的一个简单的做法便是引入momentum

2、tf.train.MomentumOptimizer 动量


3、tf.train.AdaGradOptimizer


4、tf.train.RMSPropOptimizer


5、tf.train.AdamOptimizer

'''