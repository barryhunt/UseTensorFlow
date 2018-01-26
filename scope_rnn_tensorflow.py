# -*- coding: utf-8 -*-
# /* Created by PC on 2018/1/25.*/
# 22 scope (name_scope/variable_scope)
'''
rnn中为什么重用变量？
    在 training RNN 和 test RNN 的时候, RNN 的 time_steps 会有不同的取值, 这将会影响到整个 RNN 的结构, 所以导致在 test 的时候, 不能单纯地使用 training 时建立的那个 RNN. 但是 training RNN 和 test RNN 又必须是有同样的 weights biases 的参数. 所以, 这时, 就是使用 reuse variable 的好时机
'''
import tensorflow as tf

#首先定义training 和 test 的不同参数
class TrainConfig:
    batch_size = 20
    time_steps = 20
    input_size = 10
    output_size = 2
    cell_size = 11
    learning_rate = 0.01


class TestConfig(TrainConfig):
    time_steps = 1


class RNN(object):

    def __init__(self, config):
        self._batch_size = config.batch_size
        self._time_steps = config.time_steps
        self._input_size = config.input_size
        self._output_size = config.output_size
        self._cell_size = config.cell_size
        self._lr = config.learning_rate
        self._built_RNN()

    def _built_RNN(self):
        with tf.variable_scope('inputs'):
            self._xs = tf.placeholder(tf.float32, [self._batch_size, self._time_steps, self._input_size], name='xs')
            self._ys = tf.placeholder(tf.float32, [self._batch_size, self._time_steps, self._output_size], name='ys')
        with tf.name_scope('RNN'):
            with tf.variable_scope('input_layer'):
                l_in_x = tf.reshape(self._xs, [-1, self._input_size], name='2_2D')  # (batch*n_step, in_size)
                # Ws (in_size, cell_size)
                Wi = self._weight_variable([self._input_size, self._cell_size])
                print(Wi.name)
                # bs (cell_size, )
                bi = self._bias_variable([self._cell_size, ])
                # l_in_y = (batch * n_steps, cell_size)
                with tf.name_scope('Wx_plus_b'):
                    l_in_y = tf.matmul(l_in_x, Wi) + bi
                l_in_y = tf.reshape(l_in_y, [-1, self._time_steps, self._cell_size], name='2_3D')

            with tf.variable_scope('cell'):
                cell = tf.contrib.rnn.BasicLSTMCell(self._cell_size)
                with tf.name_scope('initial_state'):
                    self._cell_initial_state = cell.zero_state(self._batch_size, dtype=tf.float32)

                self.cell_outputs = []
                cell_state = self._cell_initial_state
                for t in range(self._time_steps):
                    if t > 0: tf.get_variable_scope().reuse_variables()
                    cell_output, cell_state = cell(l_in_y[:, t, :], cell_state)
                    self.cell_outputs.append(cell_output)
                self._cell_final_state = cell_state

            with tf.variable_scope('output_layer'):
                # cell_outputs_reshaped (BATCH*TIME_STEP, CELL_SIZE)
                cell_outputs_reshaped = tf.reshape(tf.concat(self.cell_outputs, 1), [-1, self._cell_size])
                Wo = self._weight_variable((self._cell_size, self._output_size))
                bo = self._bias_variable((self._output_size,))
                product = tf.matmul(cell_outputs_reshaped, Wo) + bo
                # _pred shape (batch*time_step, output_size)
                self._pred = tf.nn.relu(product)    # for displacement

        with tf.name_scope('cost'):
            _pred = tf.reshape(self._pred, [self._batch_size, self._time_steps, self._output_size])
            mse = self.ms_error(_pred, self._ys)
            mse_ave_across_batch = tf.reduce_mean(mse, 0)
            mse_sum_across_time = tf.reduce_sum(mse_ave_across_batch, 0)
            self._cost = mse_sum_across_time
            self._cost_ave_time = self._cost / self._time_steps

        with tf.variable_scope('trian'):
            self._lr = tf.convert_to_tensor(self._lr)
            self.train_op = tf.train.AdamOptimizer(self._lr).minimize(self._cost)

    @staticmethod
    def ms_error(y_target, y_pre):
        return tf.square(tf.subtract(y_target, y_pre))

    @staticmethod
    def _weight_variable(shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=0.5, )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    @staticmethod
    def _bias_variable(shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    train_config = TrainConfig()
    test_config = TestConfig()

    # 在训练rnn过程中，错误重用参数的方法
    with tf.variable_scope('train_rnn'):
        train_rnn1 = RNN(train_config)
    with tf.variable_scope('test_rnn'):
        test_rnn1 = RNN(test_config)

    # 在训练rnn过程中，正确重用参数的方法
'''让 train_rnn 和 test_rnn 在同一个 tf.variable_scope('rnn') 之下.、
 并且定义 scope.reuse_variables(), 
 使我们能把 train_rnn 的所有 weights, biases 参数全部绑定到 test_rnn 中. 
 这样, 不管两者的 time_steps 有多不同, 结构有多不同, 
 train_rnn W, b 参数更新成什么样, test_rnn 的参数也更新成什么样.'''
    with tf.variable_scope('rnn') as scope:
        sess = tf.Session()
        train_rnn2 = RNN(train_config)
        scope.reuse_variables()
        test_rnn2 = RNN(test_config)

        init = tf.global_variables_initializer()
        sess.run(init)