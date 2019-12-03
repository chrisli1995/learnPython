import re
import codecs
import pickle
import time
import numpy as np
import tensorflow as tf
from sequence_to_sequence1 import SequenceToSequence
from data_utils import batch_flow_bucket as batch_flow
from thread_generator import ThreadedGenerator
import json

# zidian={'你':1,'好':2,'啊':3}
# lists=['你','好','啊']
# aa=[
#         ['你','好','啊'],
#         ['你','好','哦']
#     ]
#
# group=[['畹', '华', '吾', '侄'], ['你', '接', '到', '这', '封', '信', '的', '时', '候'], ['不', '知', '道', '大', '伯', '还', '在', '不', '在', '人', '世', '了']]
# for k in aa:
#     print(k)
#
# for w, in zidian:
#     print(w)
#
# print(lists[:-1])
#
# for i,line in enumerate(group):
#     print(line)
#
# line=['你', '接', '到', '这', '封','？' ,'信', '的', '时', '候']
# print(''.join(line))
# def make_split(line):
#     if re.match(r'.*([，…?!\.,!？])$', ''.join(line)):
#         return []
#     return [',']
# print(make_split(line))

# 测试pickle文件的读取
# inf=pickle.load(open('D:\\nstart\\study\\chatbot\\data\\chatbot.pkl','rb'))
# print(type(inf[0][1]),type(inf[1]))
# for i in inf[2]:(([]),([]))
#     print(i)
#     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#     time.sleep(5)
# print('xxx',inf[0][0],'xxx',inf[0][1],'xxx',inf[0][2])

# dictionary={
#         'a':'1',
#         'b': '2',
#         'c': '3',
#         'd': '4',
#         'aa': '1',
#         'bb': '2',
#         'cc': '3',
#         'dd': '4',
#         'aaa': '1',
#     }
# list = sorted(list(dictionary.keys()))
# print(list)

# print([i for i in range(10)])
# int=111
# print('type的类型为{}'.format(type(int)))
#
# encoder_state=[1,2,3,4,5,6]
# print(tuple(encoder_state))

# # 测试yield
# def gene():
#     i = 0
#     while True:
#         yield i
#         i += 1
#
# g=gene()
# print(next(g))
# print(next(g))

# 测试zip
# a = [1,2,3]
# b = [4,5,6]
# zipped=zip(a,b)
# zipped_list=list(zip(a,b))
# print(next(zipped))
# print(zipped_list)
# x_data,y_data=zip(*zipped_list)
# data=(x_data,y_data)
# print(x_data)
# print(data)
# print(*[a,b])

# 测试假数据
# dictionary = {
#     'a': '1',
#     'b': '2',
#     'c': '3',
#     'd': '4',
#     'aa': '1',
#     'bb': '2',
#     'cc': '3',
#     'dd': '4',
#     'aaa': '1',
# }
# input_list=sorted(list(dictionary.keys()))
# print(input_list)

# 测试numpy
# lengths=[1,2,3,4,5,6,7,8,9,10]
# n=np.array([1,2,3,4,5])
# print(n[[1,2,3]])
# li=np.array(lengths)[
#         (np.linspace(0,1,5,endpoint=False)*len(lengths)).astype(int)
#     ].tolist()
# print(li)

# 测试mask
# a=tf.sequence_mask([1, 3, 2], 5,dtype=tf.float32)
# b=tf.layers.Dense(32,
#                    dtype=tf.float32,
#                    use_bias=False,
#                    name='attention_cell_input_fn')
# c=b(tf.constant([[1,1]]),-1)
# with tf.Session() as sess:
#     print(sess.run(a))
#     print(type(b))
#     print(type(c))

# 测试train
# x_data,y_data=pickle.load(open('./data/chatbot.pkl','rb'))
# # print(x_data[0])
# # print(y_data[0])
# # ws=pickle.load(open('./data/ws.pkl','rb'))
# # flow=ThreadedGenerator(
# #                 batch_flow([x_data,y_data],ws,128,add_end=[False,True]),
# #                 quene_maxsize=30
# #             )
# # print(next(flow))


