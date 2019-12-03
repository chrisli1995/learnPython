import tensorflow as tf
import os
import codecs
import sys
import numpy as np
import math
import csv
from reforlstm import Relstm

# 打印日志
tf.logging.set_verbosity(tf.logging.INFO)

# 参数管理
def get_default_params():
    return tf.contrib.training.HParams(
        num_embedding_size=16,#词向量维数
        num_timesteps=50,#LSTM的步长
        num_lstm_nodes=[32,32],#LSTM的size
        num_lstm_layers=2,#LSTM的层数
        num_fc_nodes=32,#全连接层的大小
        batch_size=100,
        clip_lstm_grads=1.0,#防止梯度爆炸，设置梯度的上限
        learning_rate=0.001,
        num_word_threshold=10,#统计词表的阈值，超过这个阈值才纳入词表
        num_sentence=5
    )

# 调用参数管理的函数
hps=get_default_params()


# 文件夹定义
train_file='./data/re_output_train.csv'
val_file='./data/re_output_val.csv'
test_file='./data/re_output_test.csv'
vocab_file='./data/nlpcc_vocab.txt'
category_file='./data/nlpcc_category.txt'
output_folder='./output'

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# 字典的封装
class Vocab(object):
    def __init__(self,filname,num_word_threshold):
        self._word_to_id={}
        self._unk=-1
        self._num_word_threshold=num_word_threshold
        self._read_dict(filname)

    # 构建词汇字典,返回的是一个字典{'你':1,'我'：2}
    def _read_dict(self,filename):
        tf.logging.info('正在构建词汇表。。。')
        line_num=1
        f=codecs.open(filename,'r',encoding='utf-8')
        lines=f.readlines()

        for line in lines:
        # for i in range(77323):
            word,frequency=line.strip('\r\n').split('\t')
            frequency=int(frequency)
            if frequency<self._num_word_threshold:
                continue
            idx=len(self._word_to_id)
            if word=='<UNK>':
                self._unk=idx
            self._word_to_id[word]=idx
            # print('已处理',line_num,'行')
            # line_num+=1
        f.close()

    # 若一个词字典中没有返回<unk>
    def word_to_id(self, word):
        return self._word_to_id.get(word, self._unk)

    @property
    def unk(self):
        return self._unk
    def size(self):
        return len(self._word_to_id)

    # 将句子转换为id
    def sentence_to_id(self, sentence):
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
        return word_ids

# 测试Vocab类
# vocab=Vocab(vocab_file,hps.num_word_threshold)
# tf.logging.info('vocab_size:%d'% vocab.size())
#
# test_str='的 在 了 是'
# print(vocab.sentence_to_id(test_str))

# 类别的封装
class CategoryDict:
    def __init__(self,filename):
        self._category_to_id={}
        f=codecs.open(filename,'r',encoding='utf-8')
        lines=f.readlines()
        for line in lines:
            category=line.strip('\r\n')
            idx=len(self._category_to_id)
            self._category_to_id[category]=idx

    def category_to_id(self,category):
        if category not in self._category_to_id:
            return Exception(category,'类不是定义的标签')
        return self._category_to_id[category]

    def size(self):
        return len(self._category_to_id)

# 测试CategoryDict类
# category_vocab=CategoryDict(category_file)
# test_str='爱情'
# tf.logging.info('label:%s,id:%d' % (test_str,category_vocab.category_to_id(test_str)))

# 数据集的封装
class TextDataSet:
    def __init__(self,filename,vocab,category_vocab,num_timesteps):
        self._vocab=vocab
        self._category_vocab=category_vocab
        self._num_timesteps=num_timesteps
        # 输入的格式为[[{表达实体对1的句子1，前一句和后一句},{表达实体对1的句子2，前一句和后一句}],[{表达实体对2的句子1，前一句和后一句},{表达实体对2的句子2，前一句和后一句}]]
        self._inputs=[]
        # 输出为一个向量
        self._outputs=[]
        self._indicator=0
        self._parse_file(filename)

    # def _parse_file(self,filename):
    #     tf.logging.info('正在从%s加载数据',filename)
    #     f=codecs.open(filename,'r',encoding='utf-8')
    #     csv_reader=csv.reader(f)
    #     data_index=1
    #     data_re=[]
    #     sentence_re={}
    #     ids_label=-1
    #     for line in csv_reader:
    #         if line[0]==data_index:
    #             ids_sentence=self._vocab.sentence_to_id(line[4])[0:self._num_timesteps]
    #             padding_num = self._num_timesteps - len(ids_sentence)
    #             ids_sentence = ids_sentence + [self._vocab.unk for i in range(padding_num)]
    #             ids_next_sentence=self._vocab.sentence_to_id(line[5])[0:self._num_timesteps]
    #             padding_num = self._num_timesteps - len(ids_next_sentence)
    #             ids_next_sentence = ids_next_sentence + [self._vocab.unk for i in range(padding_num)]
    #             ids_last_sentence=self._vocab.sentence_to_id(line[6])[0:self._num_timesteps]
    #             padding_num = self._num_timesteps - len(ids_last_sentence)
    #             ids_last_sentence = ids_last_sentence + [self._vocab.unk for i in range(padding_num)]
    #             sentence_re['sentence']=ids_sentence
    #             sentence_re['next_sentence'] = ids_next_sentence
    #             sentence_re['last_sentence'] = ids_last_sentence
    #             data_re.append(sentence_re)
    #             ids_label = self._category_vocab.category_to_id(line[3])
    #         else:
    #             self._inputs.append(data_re)
    #             self._outputs.append(ids_label)
    #             data_index=line[0]
    #             data_re=[]
    #             sentence_re={}
    #             ids_sentence = self._vocab.sentence_to_id(line[4])[0:self._num_timesteps]
    #             padding_num = self._num_timesteps - len(ids_sentence)
    #             ids_sentence = ids_sentence + [self._vocab.unk for i in range(padding_num)]
    #             ids_next_sentence = self._vocab.sentence_to_id(line[5])[0:self._num_timesteps]
    #             padding_num = self._num_timesteps - len(ids_next_sentence)
    #             ids_next_sentence = ids_next_sentence + [self._vocab.unk for i in range(padding_num)]
    #             ids_last_sentence = self._vocab.sentence_to_id(line[6])[0:self._num_timesteps]
    #             padding_num = self._num_timesteps - len(ids_last_sentence)
    #             ids_last_sentence = ids_last_sentence + [self._vocab.unk for i in range(padding_num)]
    #             sentence_re['sentence'] = ids_sentence
    #             sentence_re['next_sentence'] = ids_next_sentence
    #             sentence_re['last_sentence'] = ids_last_sentence
    #             data_re.append(sentence_re)
    #     self._outputs=np.asanyarray(self._outputs,dtype=np.int32)
    #     self._random_shuffle()

    # def _parse_file(self,filename):
    #     tf.logging.info('正在从%s加载数据', filename)
    #     f=codecs.open(filename,'r',encoding='utf-8')
    #     csv_reader=csv.reader(f)
    #     data_index = 1
    #     for line in csv_reader:
    #         if int(line[0])==data_index:
    #             ids_sentence,ids_next_sentence,ids_last_sentence=self._extract_inf(line)
    #             self._inputs.append(ids_sentence+ids_next_sentence+ids_last_sentence)
    #             ids_label = self._category_vocab.category_to_id(line[3])
    #             self._outputs.append(ids_label)
    #             data_index+=1
    #         else:
    #             continue
    #     self._inputs = np.asarray(self._inputs, dtype=np.int32)
    #     self._outputs = np.asarray(self._outputs, dtype=np.int32)
    #     self._random_shuffle()
    def _extract_inf(self,line):
        ids_sentence = self._vocab.sentence_to_id(line[4])[0:self._num_timesteps]
        padding_num = self._num_timesteps - len(ids_sentence)
        ids_sentence = ids_sentence + [self._vocab.unk for i in range(padding_num)]
        # ids_next_sentence = self._vocab.sentence_to_id(line[5])[0:self._num_timesteps]
        # padding_num = self._num_timesteps - len(ids_next_sentence)
        # ids_next_sentence = ids_next_sentence + [self._vocab.unk for i in range(padding_num)]
        # ids_last_sentence = self._vocab.sentence_to_id(line[6])[0:self._num_timesteps]
        # padding_num = self._num_timesteps - len(ids_last_sentence)
        # ids_last_sentence = ids_last_sentence + [self._vocab.unk for i in range(padding_num)]
        return  ids_sentence

    def _open_csv(self,filename):
        f = codecs.open(filename, 'r', encoding='utf-8')
        csv_reader = csv.reader(f)
        return csv_reader

    def _line_num(self,filename):
        f = codecs.open(filename, 'r', encoding='utf-8')
        csv_reader = csv.reader(f)
        line_num=0
        data=[]
        for line in csv_reader:
            line_num+=1
            data.append(line[0])
        return line_num,data[-1]

    def _parse_file(self, filename):
        tf.logging.info('正在从%s加载数据', filename)
        # f = codecs.open(filename, 'r', encoding='utf-8')
        # csv_reader = csv.reader(f)
        line_num,data_last_index=self._line_num(filename)
        # print(data_last_index)
        data_index = 1
        line_index=[]
        index_index=0
        re_1 = []
        re_2 = []
        re_3 = []
        re_4 = []
        re_5 = []
        for i,line in enumerate(self._open_csv(filename)):
            if int(line[0])==data_index:
                line_index.append(i)
                data_index+=1
        line_index.append(-1)
        # print('index:',line_index)
        data_index=1
        for k,line in enumerate(self._open_csv(filename)):
            if k==int(line_index[index_index]):
                ids_label = self._category_vocab.category_to_id(line[3])
                self._outputs.append(ids_label)
                index_index+=1
        index_index=0
        for j,line in enumerate(self._open_csv(filename)):
            if j==line_index[index_index] and data_index==int(line[0]):
                ids_sentence = self._extract_inf(line)
                re_1.append(ids_sentence)
                data_index += 1
                index_index+=1
            elif j==line_index[index_index] and data_index!=int(line[0]):
                re_1.append([0 for _ in range(50)])
                data_index+=1
                index_index+=1
        index_index = 0
        data_index = 1
        for j,line in enumerate(self._open_csv(filename)):
            if int(data_index) == int(data_last_index) + 1:
                break
            if j==line_index[index_index]+1 and data_index==int(line[0]):
                # print(j, line_index[index_index], data_index, line[0])
                ids_sentence = self._extract_inf(line)
                re_2.append(ids_sentence)
                data_index += 1
                index_index+=1
            elif (j==line_index[index_index]+1 and data_index!=int(line[0])) or j+1>line_num:
                re_2.append([0 for _ in range(50)])
                data_index += 1
                index_index += 1
        index_index = 0
        data_index = 1
        for j,line in enumerate(self._open_csv(filename)):
            if int(data_index) == int(data_last_index) + 1:
                break
            elif j==line_index[index_index]+2 and data_index==int(line[0]):
                ids_sentence = self._extract_inf(line)
                re_3.append(ids_sentence)
                data_index += 1
                index_index+=1
            elif (j==line_index[index_index]+2 and data_index!=int(line[0])) or j+2>line_num:
                re_3.append([0 for _ in range(50)])
                data_index += 1
                index_index += 1
        index_index = 0
        data_index = 1
        for j,line in enumerate(self._open_csv(filename)):
            if int(data_index) == int(data_last_index) + 1:
                break
            elif j==line_index[index_index]+3 and data_index==int(line[0]):
                ids_sentence = self._extract_inf(line)
                re_4.append(ids_sentence)
                data_index += 1
                index_index+=1
            elif (j==line_index[index_index]+3 and data_index!=int(line[0])) or j+3>line_num:
                re_4.append([0 for _ in range(50)])
                data_index += 1
                index_index += 1
        index_index = 0
        data_index = 1
        for j,line in enumerate(self._open_csv(filename)):
            if int(data_index)==int(data_last_index)+1:
                break
            elif j==line_index[index_index]+4 and data_index==int(line[0]):
                ids_sentence = self._extract_inf(line)
                re_5.append(ids_sentence)
                data_index += 1
                index_index+=1
            elif (j==line_index[index_index]+4 and data_index!=int(line[0])) or j+4>line_num:
                re_5.append([0 for _ in range(50)])
                data_index += 1
                index_index += 1
        print('r1:',len(re_1),'r2:',len(re_2),'r3:',len(re_3),'r4:',len(re_4),'r5:',len(re_5))
        self._inputs.append(re_1)
        self._inputs.append(re_2)
        self._inputs.append(re_3)
        self._inputs.append(re_4)
        self._inputs.append(re_5)
        self._inputs = np.asarray(self._inputs, dtype=np.int32)
        self._outputs = np.asarray(self._outputs, dtype=np.int32)
        print('inputs:',self._inputs.shape)
        print('outputs:',self._outputs.shape)
        self._random_shuffle()


    def _list_permutation(self,list,permutation):
        new_list=[]
        for i in permutation:
            new_list.append(list[i])
        return new_list

    def _random_shuffle(self):
        p=np.random.permutation(len(self._inputs[1,:,:]))
        # tem_arr=np.array([],dtype=np.int32)
        tem_arr=[]
        for arr in self._inputs.tolist():
            arr_new = self._list_permutation(arr,p)
            tem_arr.append(arr_new)
        self._inputs=np.asarray(tem_arr, dtype=np.int32)
        self._outputs=self._outputs[p]

    def size(self):
        return len(self._inputs[1])

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > len(self._inputs[1]):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        if end_indicator > len(self._inputs[1]):
            raise Exception('batch_size: %d 太大了' % batch_size)
        tem_arr=[]
        for arr in self._inputs.tolist():
            arr_new = arr[self._indicator:end_indicator]
            tem_arr.append(arr_new)
        batch_inputs=np.asarray(tem_arr, dtype=np.int32)
        batch_outputs = self._outputs[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_inputs, batch_outputs

# 测试TextDataSet类
vocab=Vocab(vocab_file,hps.num_word_threshold)
category_vocab = CategoryDict(category_file)
train_dataset=TextDataSet(train_file,vocab,category_vocab,hps.num_timesteps)
batch_inputs,batch_outputs=train_dataset.next_batch(100)
print(batch_inputs.shape,type(batch_inputs))
print(batch_outputs.shape,type(batch_outputs))
print(batch_outputs)

# 构建图模型
def create_model(hps,vocab_size,num_classes,paper=True):
    relstm=Relstm()
    num_timesteps=hps.num_timesteps
    batch_size=hps.batch_size
    # 定义输入与输出
    inputs = tf.placeholder(tf.int32, (5,batch_size, num_timesteps))
    outputs = tf.placeholder(tf.int32, (batch_size,))
    # 定义dropout，保存下来的值
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # 保存训练步数
    global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step', trainable=False)

    # 句子编码器
    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope('embedding',initializer=embedding_initializer):
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
    # with tf.variable_scope('lstm_nn'):
    #     cells = []
    #     for i in range(hps.num_lstm_layers):
    #         cell = tf.contrib.rnn.BasicLSTMCell(
    #             hps.num_lstm_nodes[i],
    #             state_is_tuple=True,
    #             name='lstm-'+str(i)
    #         )
    #         cell = tf.contrib.rnn.DropoutWrapper(
    #             cell,
    #             output_keep_prob=keep_prob)
    #         cells.append(cell)
    #     cell = tf.contrib.rnn.MultiRNNCell(cells)
    #     last_1 = lstm_encoder(embed_inputs_1,cell,batch_size)
    #     last_2 = lstm_encoder(embed_inputs_2, cell, batch_size)
    #     last_3 = lstm_encoder(embed_inputs_3, cell, batch_size)
    #     last_4 = lstm_encoder(embed_inputs_4, cell, batch_size)
    #     last_5 = lstm_encoder(embed_inputs_5, cell, batch_size)
    cell=relstm.multi_lstm(hps=hps, keep_prob=keep_prob)
    last_1=relstm.lstm_encoder(embed_inputs_1, hps, cell)
    last_2 = relstm.lstm_encoder(embed_inputs_2, hps, cell)
    last_3 = relstm.lstm_encoder(embed_inputs_3, hps, cell)
    last_4 = relstm.lstm_encoder(embed_inputs_4, hps, cell)
    last_5 = relstm.lstm_encoder(embed_inputs_5, hps, cell)
    # 这里的last在第二维上（可以理解为列）取最后一个数据，维数为[batch_size,lstm_output[-1](32)]
    # 通过注意力机制计算出每一组输入的权重，这里的weight为[batch_size,5]
    weight=relstm.attention_re(last_1, last_2, last_3, last_4, last_5, outputs)
    weight_1,weight_2,weight_3,weight_4,weight_5=tf.split(weight,axis=1,num_or_size_splits=5)
    last=tf.multiply(last_1,weight_1)+tf.multiply(last_2,weight_2)+tf.multiply(last_3,weight_3)+tf.multiply(last_4,weight_4)+tf.multiply(last_5,weight_5)
    # 加入注意力机制后的last纬度并没有改变依然是[batch_size,lstm_output[-1](32)]

    # 绘制散点图
    # scatter(last)

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

    # 计算损失函数
    with tf.name_scope('metrics'):
        # sparse_softmax_cross_entropy_with_logits将logits先softmax以后和做了one-hot的label做cross entropy
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=outputs
        )
        # loss的结果为[batch_size,1]
        loss = tf.reduce_mean(softmax_loss)
        # [0,1,5,4,2]->argmax:2
        y_pred = tf.argmax(tf.nn.softmax(logits), 1, output_type=tf.int32)
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

if __name__ == '__main__':
    # 训练过程
    # 创建词表
    vocab = Vocab(vocab_file, hps.num_word_threshold)
    vocab_size=vocab.size()
    print('vocab_size:',vocab_size)
    # 创建类别
    category_vocab = CategoryDict(category_file)
    num_classes = category_vocab.size()
    # 数据集封装
    train_dataset=TextDataSet(train_file,vocab,category_vocab,hps.num_timesteps)
    test_dataset=TextDataSet(test_file,vocab,category_vocab,hps.num_timesteps)
    # 查看网络
    placeholders,metrics,others,last=create_model(hps,vocab_size,num_classes)
    inputs,outputs,keep_prob=placeholders
    loss,accuracy=metrics
    train_op,global_step=others
    # 定义一些参数
    init_op = tf.global_variables_initializer()
    train_keep_prob_value = 0.8
    test_keep_prob_value = 1.0
    num_train_steps = 3000

    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(num_train_steps):
            batch_inputs, batch_labels = train_dataset.next_batch(hps.batch_size)
            outputs_val = sess.run([loss, accuracy, train_op, global_step],
                                   feed_dict={
                                       inputs: batch_inputs,
                                       outputs: batch_labels,
                                       keep_prob: train_keep_prob_value
                                   })
            loss_val, accuracy_val, _, global_step_val = outputs_val
            if global_step_val % 100 == 0:
                tf.logging.info('step:%5d,loss:%3.3f,accuracy:%3.3f' % (global_step_val, loss_val, accuracy_val))
            if i==num_train_steps-1:
                test_inputs,test_labels=test_dataset.next_batch(hps.batch_size)
                outputs_test=sess.run([loss, accuracy,last],
                                      feed_dict={
                                          inputs:test_inputs,
                                          outputs:test_labels,
                                          keep_prob:train_keep_prob_value
                                      })
                loss_test,accuracy_test,last_val=outputs_test
                tf.logging.info('[test_date] loss:%3.3f,accuracy:%3.3f' % (loss_test,accuracy_test))
                saver.save(sess, './output/lstm+att/model')
