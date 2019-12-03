import codecs
import json
import ijson
import re
import jieba
import os
import csv
import tensorflow as tf
import numpy as np
import inspect
from reforlstm import Relstm
import math
from urllib.parse import quote
import urllib


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

# f=codecs.open('./data/re.json','r')
# line=f.readline(0)
# json_file=json.load(f)
# print(type(line))
# print(type(json_file))
# print(json_file[0])
# print(type(json_file[0]))
# print(len(json_file))
# print(json_file[0]['r'])
# for data in json_file:
#     if data['1']=='谢娜':
#         print(data)
# entiy1='王珂'
# entiy2='李宇春'
# test=' 有 记者 前日 联系 到 天娱 的 副总 王珂 ， 他 表示 ， 目前 李宇春 在 《 十月 围城 》 的 角色 还 没有 确定 ， 之前 有 消息 说 她 演王 学圻 的 女儿 ， 但是 此次 到 上海 去 试镜 只是 试 了 几套 服装 ， 和 表演 方面 的 内容 完全 无关 '
# if re.findall(r'%s'% entiy1,test) and re.findall(r'%s'% entiy2 ,test):
#     print('yes')
# else:
#     print('no')

# 测试读取json文件
# f=codecs.open('./data/datasets.json','r',encoding='utf-8')
# content=f.readline()
# print(type(content))
# json_file=json.loads(content)
# print(len(json_file))

# with codecs.open('./data/re_output.json','r',encoding='utf-8') as f:
#     objects=ijson.items(f,'item')
#     print(next(objects))
#     # for object in objects:
#     #     print(object)

# 测试正则表达式
# f=codecs.open('./data/news_train_fc.txt','r',encoding='utf-8')
# line=f.readline()
# line_num=0
# while line:
#     # print(line)
#     if re.findall(r'/ 文*',line):
#         print('catch')
#     line=f.readline()
#     line_num+=1
#     if line_num==20:
#         break


# 测试消除停用词
# f=codecs.open('./data/停用词整合.txt','r',encoding='utf-8')
# sw=f.readlines()
# stopwords = []
# for stline in sw:
#     stline = stline.strip()
#     stopwords.append(stline)
# print(stopwords)
# f1=codecs.open('./data/11.txt','r',encoding='utf-8')
# line=f1.readline()
# while line:
#     seg_line=''
#     words=jieba.cut(line)
#     for word in words:
#         # print(word)
#         if word not in stopwords:
#             seg_line=seg_line+word+' '
#     print(seg_line)
#     line=f1.readline()

# 测试读取多文件
# rootdir = 'D:\\data\\news\\THUCNews\\体育'
# f_w=codecs.open('./data/nlpcc_fc.txt','a',encoding='utf-8')
# dirlist = os.listdir(rootdir) #列出文件夹下所有的目录与文件
# stopwords=[]
# sw = codecs.open('./data/停用词整合1.txt', 'rb', encoding='utf8')
# fsw=sw.readlines()
# for stline in fsw:
#     stline = stline.strip()
#     stopwords.append(stline)
#
# file_num=0
# for i in range(0,len(dirlist)):
#     path = os.path.join(rootdir,dirlist[i])
#     if os.path.isfile(path):
#         print(path)
#         f=codecs.open(path,'r',encoding='utf-8')
#         line=f.readline()
#         while line:
#             words=jieba.cut(line)
#             line_seg = ''
#             for word in words:
#                 if word not in stopwords:
#                     line_seg += word + " "
#             sentences = re.split('[。？！(。”)]', line_seg)
#             for sentence in sentences:
#                 sentence = sentence.strip()
#                 f_w.writelines(sentence + '\n')
#             line=f.readline()
#         file_num+=1
#         print('已完成',file_num,'个文件')
#         if file_num==10000:
#             break

# 测试CSV文件的读取写入
# csv_file=csv.reader(open('./data/re_output_train.csv','r',encoding='utf-8'))
# print(type(csv_file))
# for i,stu in enumerate(csv_file):
#     print(type(i),'-----',stu)
#     print(stu[4:])
# # csv_file=csv.reader(codecs.open('./data/re_output_train.csv','r',encoding='utf-8'))
# for i,stu in enumerate(csv_file):
#     print('xxxxxxxxxxx')
# csv_file_test=csv.writer(open('./data/re_output_train.csv','a',newline='',encoding='utf-8'),dialect='excel')
# for i,stu in enumerate(csv_file):
#     print(stu)
#     if i>1000:
#         csv_file_test.writerow(stu)

# 测试随机函数
# p=np.random.permutation(10)
# print(type(p[0]))

# 统计每个关系有几个句子表示
# csv_file=csv.reader(codecs.open('./data/re_output_train.csv','r',encoding='utf-8'))
# print(type(csv_file))
# test={}
# for i,stu in enumerate(csv_file):
#     test.setdefault(stu[0],0)
#     test[stu[0]]+=1
# print(test)
# sum=0
# num=0
# for t1 in test:
#     sum+=test[t1]
#     num+=1
# print(sum/num)

# 关于np的一些测试
array=np.asanyarray([[[1,1],[2,2]],
                     [[3,3],[4,4]],
                     [[5, 5], [6, 6]]])
array1=np.asanyarray([[1,2],
                     [3,4]])
print(np.concatenate(array,axis=1))
print(np.reshape(np.concatenate(array,axis=1),(-1,2)))
print(np.concatenate(array1,axis=0))
# array1=np.array(np.arange(12).reshape(3,2,2))
# print(type(array))
# print(len(array[:,1,:]))
# num=1
# for list in array:
#     np.insert(array1,num,list,0)
#     num+=1
# print(array1.shape)

# 动态获取变量
# def get_variable_name(variable):
#     callers_local_vars = inspect.currentframe().f_back.f_locals.items()
#     return [var_name for var_name, var_val in callers_local_vars if var_val is variable]
# prepare_list = locals()
# for i in range(16):
#     prepare_list['list_' + str(i)] = []
#     prepare_list['list_' + str(i)].append(('我是第' + str(i)) + '个list')
# a = get_variable_name(prepare_list['list_0']).pop()
# b = get_variable_name(prepare_list['list_1']).pop()
# print(a)
# print(b)
# print(prepare_list['list_' + str(1)])

# # TensorFlow测试
# test0=np.asanyarray([[[1,1],[2,2]],
#                      [[3,3],[4,4]],
#                      [[5, 5], [6, 6]]])
# result0=tf.concat(test0,1)
# test1=np.array([[1,0,2,1,0]],dtype=np.float32)
# test2=np.array([[0.1],
#                [0.2],
#                [0.3],
#                [0.4],
#                [0.5],],dtype=np.float32)
# test3=np.array([[1,0,2,1,0]],dtype=np.float32)
# test4=np.array([[6,3,5],
#                 [2,4,6]],dtype=np.float32)
# test5=np.array([[1,2],
#                 [3,4]],dtype=np.float32)
# test6=tf.constant([[1,2],
#                 [3,4]])
# result1=tf.matmul(test1,test2)
# result2=tf.concat([test1,test3],axis=0)
# result3=tf.nn.softmax(test4,axis=0)
# w1,w2,w3=tf.split(test4,num_or_size_splits=3,axis=1)
# result4=tf.multiply(test5,w1)
# result5=tf.reduce_sum(test5,axis=1,keep_dims=True)
# result6=test1+test3
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(result0))
#     onehot=sess.run(result5)
#     re=sess.run(result3)
#     print(test6.eval())
#     print(onehot)

# global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step', trainable=False)
# attention_weight=tf.get_variable('attention_weight',(32,3),tf.float32)
# keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# cell=tf.contrib.rnn.BasicLSTMCell(
#             32,
#             state_is_tuple=True,
#             name='lstm-0'
#         )
# with tf.variable_scope('embedding'):
#     embeddings = tf.get_variable('embedding', [28471, 16], tf.float32)
# relstm=Relstm()
# relstm.multi_lstm(hps,keep_prob)
# new_saver = tf.train.Saver()
# with tf.Session() as sess:
#   new_saver.restore(sess, './output_non/model')
#   print(sess.run(global_step))
#   print(sess.run(attention_weight))
#   print(sess.run(relstm.cell))
  # my_saver = tf.train.import_meta_graph('./output_non/model.meta')
  # my_saver.restore(sess, tf.train.latest_checkpoint('./output_non'))
  # graph=tf.get_default_graph()
  # print(graph.get_tensor_by_name('global_step'))

# # 测试百度知识图谱
# ch_str = quote('林俊杰')
# en_url = 'http://shuyantech.com/api/cndbpedia/avpair?q='
# url = en_url + ch_str
# response2 = urllib.request.urlopen(url)
# print(response2.read().decode('utf-8'))



