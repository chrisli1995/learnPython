import tensorflow as tf
import numpy as np
import math

class Relstm:
    def lstm_result(self,embed_inputs, cell, batch_size,test=False):
        initial_state = cell.zero_state(batch_size, tf.float32)
        # 输出结果的格式：[batch_size,3*num_timesteps（150）,lstm_outputs[-1]（32）]
        rnn_outputs, _ = tf.nn.dynamic_rnn(cell,
                                           embed_inputs,
                                           initial_state=initial_state)
        if test == True:
            rnn_outputs=tf.layers.batch_normalization(rnn_outputs,training=False)
        else:
            rnn_outputs = tf.layers.batch_normalization(rnn_outputs, training=True)
        last = rnn_outputs[:, -1, :]
        return last

    # 加入attention机制，返回每个输入的权重
    def attention_re(self,last1, last2, last3, last4, last5, outputs):
        # 输入的last的形状为[batch_size,lstm_output[-1](32)]
        attention_weight = tf.get_variable('attention_weight', (last1.shape[1], 3), tf.float32)
        # 这里将权重weight转换为[batch_size,3]，与one-hot过后的分类结果进行点乘，并按列相加
        weight_1 = tf.matmul(last1, attention_weight)
        weight_2 = tf.matmul(last2, attention_weight)
        weight_3 = tf.matmul(last3, attention_weight)
        weight_4 = tf.matmul(last4, attention_weight)
        weight_5 = tf.matmul(last5, attention_weight)
        # target=tf.one_hot(outputs,3)
        # target = tf.get_variable('target', (1, 3), tf.float32)
        target=outputs
        # 此时的weight转为[batch_size,1]，之后进行tanh
        weight_1 = tf.tanh(tf.reduce_sum(tf.multiply(weight_1, target), axis=1, keep_dims=True))
        weight_2 = tf.tanh(tf.reduce_sum(tf.multiply(weight_2, target), axis=1, keep_dims=True))
        weight_3 = tf.tanh(tf.reduce_sum(tf.multiply(weight_3, target), axis=1, keep_dims=True))
        weight_4 = tf.tanh(tf.reduce_sum(tf.multiply(weight_4, target), axis=1, keep_dims=True))
        weight_5 = tf.tanh(tf.reduce_sum(tf.multiply(weight_5, target), axis=1, keep_dims=True))
        print('weight_1_new:', weight_1.shape)
        weight = tf.concat([weight_1, weight_2, weight_3, weight_4, weight_5], axis=1)
        weight = tf.nn.softmax(weight, axis=1)
        return weight

    def multi_lstm(self,hps,keep_prob,test=False):
        scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_lstm_nodes[-1]) / 3.0
        lstm_init = tf.random_uniform_initializer(-scale, scale)
        if test==True:
            with tf.variable_scope('lstm_nn'):
                cells = []
                for i in range(hps.num_lstm_layers):
                    cell = tf.contrib.rnn.BasicLSTMCell(
                        hps.num_lstm_nodes[i],
                        state_is_tuple=True,
                        name='lstm-' + str(i)
                    )
                    cell = tf.contrib.rnn.DropoutWrapper(
                        cell,
                        output_keep_prob=keep_prob)
                    cells.append(cell)
                cell = tf.contrib.rnn.MultiRNNCell(cells)
                self.cell=cell
                return cell
        else:
            with tf.variable_scope('lstm_nn',initializer=lstm_init):
                cells = []
                for i in range(hps.num_lstm_layers):
                    cell = tf.contrib.rnn.BasicLSTMCell(
                        hps.num_lstm_nodes[i],
                        state_is_tuple=True,
                        name='lstm-' + str(i)
                    )
                    cell = tf.contrib.rnn.DropoutWrapper(
                        cell,
                        output_keep_prob=keep_prob)
                    cells.append(cell)
                cell = tf.contrib.rnn.MultiRNNCell(cells)
                self.cell=cell
                return cell

    def lstm_encoder(self,input,hps,cell,test=False):
        last = self.lstm_result(input, cell, hps.batch_size,test)
        self.last=last
        return last

