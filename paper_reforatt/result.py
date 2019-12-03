import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import math
from reextraction import Vocab
from reextraction import CategoryDict
from reextraction import TextDataSet
from reforlstm import Relstm

# 参数管理
def get_default_params():
    return tf.contrib.training.HParams(
        num_embedding_size=16,#词向量维数
        num_timesteps=50,#LSTM的步长
        num_lstm_nodes=[32,32],#LSTM的size
        num_lstm_layers=2,#LSTM的层数
        num_fc_nodes=32,#全连接层的大小
        batch_size=36,
        clip_lstm_grads=1.0,#防止梯度爆炸，设置梯度的上限
        learning_rate=0.001,
        num_word_threshold=10,#统计词表的阈值，超过这个阈值才纳入词表
        num_sentence=5
    )
# 调用参数管理的函数
hps=get_default_params()

# 构建图模型
def test_model(hps,vocab_size,num_classes,paper=True):
    relstm=Relstm()
    num_timesteps=hps.num_timesteps
    batch_size=hps.batch_size
    # 定义输入与输出
    inputs = tf.placeholder(tf.int32, (5,batch_size, 3*num_timesteps))
    outputs = tf.placeholder(tf.int32, (batch_size,))
    # 定义dropout，保存下来的值
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # 保存训练步数
    global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step', trainable=False)

    # 句子编码器
    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope('embedding', initializer=embedding_initializer):
        # embeddings相当于单词表
        embeddings = tf.get_variable('embedding', [vocab_size, hps.num_embedding_size], tf.float32)  # 用法跟variable比较类似
        # [1,10,7]->[embeddings[1],embeddings[10],embeddings[7]],look_up相当于一个“拼接”，比如input为（3*2）,词典为（5*2）,那么look_up以后为（3*2*2）
        # 本次实验的embed_inputs为三维的[batch_size,3*num_timesteps（150）,hps.num_embedding_size]
        for i in range(hps.num_sentence):
            if i==0:
                embed_inputs_1 = tf.nn.embedding_lookup(embeddings, inputs[i])
            elif i==1:
                embed_inputs_2 = tf.nn.embedding_lookup(embeddings, inputs[i])
            elif i==2:
                embed_inputs_3 = tf.nn.embedding_lookup(embeddings, inputs[i])
            elif i==3:
                embed_inputs_4 = tf.nn.embedding_lookup(embeddings, inputs[i])
            elif i==4:
                embed_inputs_5 = tf.nn.embedding_lookup(embeddings, inputs[i])
    scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_lstm_nodes[-1]) / 3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)
    cell=relstm.multi_lstm(hps,keep_prob,test=True)
    last_1 = relstm.lstm_encoder(embed_inputs_1, hps, cell,test=True)
    last_2 = relstm.lstm_encoder(embed_inputs_2, hps, cell,test=True)
    last_3 = relstm.lstm_encoder(embed_inputs_3, hps, cell,test=True)
    last_4 = relstm.lstm_encoder(embed_inputs_4, hps, cell,test=True)
    last_5 = relstm.lstm_encoder(embed_inputs_5, hps, cell,test=True)
    # 这里的last在第二维上（可以理解为列）取最后一个数据，维数为[batch_size,lstm_output[-1](32)]
    last=last_1+last_2+last_3+last_4+last_5
    # weight=relstm.attention_re(last_1,last_2,last_3,last_4,last_5,outputs)
    # weight_1,weight_2,weight_3,weight_4,weight_5=tf.split(weight,axis=1,num_or_size_splits=5)
    # last=tf.multiply(last_1,weight_1)+tf.multiply(last_2,weight_2)+tf.multiply(last_3,weight_3)+tf.multiply(last_4,weight_4)+tf.multiply(last_5,weight_5)

    # 分类器（这里只多填了个全连接层），分类之后的结果为[batch_size,num_classes(3)]
    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer=fc_init):
        fc1 = tf.layers.dense(last,
                              hps.num_fc_nodes,
                              activation=tf.nn.relu,
                              name='fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        logits = tf.layers.dense(fc1_dropout,
                                 num_classes,
                                 name='fc2')
    # 此时的logits为一个[batch_size,last_dese_size(3)]

    # 通过注意力机制计算出每一组输入的权重，这里的weight为[batch_size,5]
    weight = relstm.attention_re(last_1, last_2, last_3, last_4, last_5, logits)
    weight_1, weight_2, weight_3, weight_4, weight_5 = tf.split(weight, axis=1, num_or_size_splits=5)
    last_last = tf.multiply(last_1, weight_1) + tf.multiply(last_2, weight_2) + tf.multiply(last_3, weight_3) + tf.multiply(last_4, weight_4) + tf.multiply(last_5, weight_5)
    # 根据加入注意力机制后的输入算出最终预测结果logits_last
    with tf.variable_scope('fc_last', initializer=fc_init):
        fc3 = tf.layers.dense(last_last,
                              hps.num_fc_nodes,
                              activation=tf.nn.relu,
                              name='fc3')
        fc3_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        logits_last = tf.layers.dense(fc3_dropout,
                                 num_classes,
                                 name='fc4')

    # 计算损失函数
    with tf.name_scope('metrics'):
        # sparse_softmax_cross_entropy_with_logits将logits先softmax以后和做了one-hot的label做cross entropy
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_last,
            labels=outputs
        )
        # loss的结果为[batch_size,1]
        loss = tf.reduce_mean(softmax_loss)
        # [0,1,5,4,2]->argmax:2
        y_pred = tf.argmax(tf.nn.softmax(logits_last), 1, output_type=tf.int32)
        correct_pred = tf.equal(outputs, y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('train_op'):
        tvars=tf.trainable_variables()
        for var in tvars:
            tf.logging.info('variable name:%s'%(var.name))
        grads,_=tf.clip_by_global_norm(tf.gradients(loss,tvars),
                                       hps.clip_lstm_grads)
        optimizer=tf.train.AdamOptimizer(hps.learning_rate)
        train_op=optimizer.apply_gradients(zip(grads,tvars),
                                           global_step=global_step)

    return ((inputs,outputs,keep_prob),
            (loss,accuracy),
            (train_op,global_step),
            last)
# 创建词表
vocab = Vocab('./data/nlpcc_vocab.txt', hps.num_word_threshold)
vocab_size=vocab.size()
print('vocab_size:',vocab_size)
# 创建类别
category_vocab = CategoryDict('./data/nlpcc_category.txt')
num_classes = category_vocab.size()
test_dataset=TextDataSet('./data/re_output_val.csv',vocab,category_vocab,hps.num_timesteps)
placeholders, metrics, others, last = test_model(hps, vocab_size, num_classes)
inputs, outputs, keep_prob = placeholders
loss, accuracy = metrics
train_op, global_step = others
test_keep_prob_value = 1.0
saver=tf.train.Saver()

with tf.Session() as sess:
    batch_inputs, batch_labels = test_dataset.next_batch(hps.batch_size)
    saver.restore(sess,'./output/lstm_fbatt/lstm+fbatt')
    outputs_val=sess.run([loss,accuracy],
             feed_dict={
                 inputs: batch_inputs,
                 outputs: batch_labels,
                 keep_prob: test_keep_prob_value
             })
    loss_val,accuracy_val=outputs_val
    tf.logging.info('loss:%3.3f,accuracy:%3.3f' % (loss_val, accuracy_val))




