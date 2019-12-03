# -*- coding: utf-8 -*-
# 需要做的处理如下
# 分词
# 词语转换为向量
#   matrix->[|V|,embed_size]
#   词表
# label转换为向量

import sys
import os
import codecs
import jieba
import re
import json
import csv

#停用词表
stopword='./data/停用词整合.txt'

train_file='./data/re_output_train.csv'
val_file='./text_classification_data/cnews.val.txt'
test_file='./text_classification_data/cnews.test.txt'

#分词后的结果
seg_train_file='./data/nlpcc_fc.txt'
seg_val_file='./text_classification_data/cnews.valfc.txt'
seg_test_file='./text_classification_data/cnews.testfc.txt'

#词表
vocab_file='./data/nlpcc_vocab.txt'
#标签表
category_file='./data/nlpcc_category.txt'
#关系
re_file='./data/re.json'
#输出的关系对齐文件
re_output_file='./data/re_output.json'


def generate_fenci_file(input_file,out_fc_file):
    line_num=0
    f=codecs.open(input_file,'r',encoding='utf-8')
    line=f.readline()
    st=codecs.open(out_fc_file,'w',encoding='utf-8')
    while line:
        content=line
        fcword=jieba.cut(content)
        word_content=''
        #去掉空格
        for word in fcword:
            word=word.strip(' ')
            if word!='':
                word_content+=word+' '
        out_line='%s\n'%(word_content.strip(' '))
        # sentences=out_line.split('。')
        sentences=re.split('[。？！]',out_line)
        for sentence in sentences:
            if not re.findall(r'/ 文*',sentence):
                st.writelines(sentence+'\n')
        # st.writelines(out_line)
        print('---------------写入第',line_num,'行-----------------')
        line_num+=1
        line=f.readline()

# 在分词的基础上加入删除停用词
def generate_fenci_file_new(input_file,out_fc_file):
    line_num=0
    f=codecs.open(input_file,'r',encoding='utf-8')
    line=f.readline()
    st=codecs.open(out_fc_file,'w',encoding='utf-8')
    stopwords = []
    sw = codecs.open('./data/停用词整合1.txt', 'rb', encoding='utf8')
    fsw=sw.readlines()
    for stline in fsw:
        stline = stline.strip()
        stopwords.append(stline)
    while line:
        words = jieba.cut(line)
        line_seg = ''
        for word in words:
            if word not in stopwords:
                line_seg += word + " "
        # sentences=out_line.split('。')
        sentences=re.split('[。？！(。”)]',line_seg)
        for sentence in sentences:
            if not re.findall(r'/ 文*',sentence):
                sentence=sentence.strip()
                st.writelines(sentence+'\n')
        # st.writelines(out_line)
        print('---------------写入第',line_num,'行-----------------')
        line_num+=1
        line=f.readline()

#统计出词表
def generate_vocab_file(input_seg_file,output_vocab_file):
    f=codecs.open(input_seg_file,'r',encoding='utf-8')
    line=f.readline()
    word_dict={}
    while line:
        print('统计词频中...')
        content = line.strip('\r\n')
        for word in content.split():
            word_dict.setdefault(word,0)
            word_dict[word]+=1
        line=f.readline()
    #排序，词频高的放上面
    sorted_word_dict=sorted(word_dict.items(),key=lambda d:d[1],reverse=True)
    st=codecs.open(output_vocab_file,'a+',encoding='utf-8')
    st.writelines('<UNK>\t1000000\n')
    print('xxxxxx')
    for item in sorted_word_dict:
        print('写入词表中...')
        st.writelines(str(item[0])+'\t'+str(item[1])+'\n')

#统计出分类结果词表
def generaete_category_dict(input_file,category_file):
    f = codecs.open(input_file, 'r', encoding='utf-8')
    f_reader=csv.reader(f)
    category_dict = {}
    for stu in f_reader:
        label = stu[3]
        category_dict.setdefault(label,0)
        category_dict[label]+=1
    category_number=len(category_dict)
    st = codecs.open(category_file, 'a+', encoding='utf-8')
    for category in category_dict:
        line='%s\n'%category
        print(category,category_dict[category])
        st.writelines(line)

# 数据与关系进行对齐
def data_match_re(data_file,re_file,output_file_csv,output_file=None):
    f_re = codecs.open(re_file, 'r',encoding='utf-8')
    f_data=codecs.open(data_file,'r',encoding='utf-8')
    if output_file==None:
        pass
    else:
        f_output=codecs.open(output_file,'a',encoding='utf-8')
    f_output_csv=open(output_file_csv,'a',newline='',encoding='utf-8')
    csv_write=csv.writer(f_output_csv,dialect='excel')
    line = f_data.readline()
    news_data=[]
    output=[]
    re_num=0
    print('正在读取数据。。。')
    while line:
        line=line.strip()
        news_data.append(line)
        line=f_data.readline()
    json_file = json.loads(f_re.readline())
    print('对齐数据中。。。')
    for guanxi in json_file:
        entity_1=guanxi['1']
        entity_2=guanxi['2']
        relation=guanxi['r']
        for i,data in enumerate(news_data):
            last_data=None
            if i>0:
                last_data=news_data[i-1]
            if i<len(news_data)-1:
                next_data=news_data[i+1]
            else:
                next_data=None
            if re.findall(r'%s'% entity_1,data) and re.findall(r'%s'% entity_2 ,data):
                output_csv = []
                print(i)
                dict_data={}
                dict_data['entity_1'] = entity_1
                dict_data['entity_2'] = entity_2
                dict_data['relation'] = relation
                dict_data['sentence'] = data
                dict_data['last_sentence']=last_data
                dict_data['next_sentence']=next_data
                print(dict_data)
                output.append(dict_data)
                output_csv.append(str(entity_1))
                output_csv.append(str(entity_2))
                output_csv.append(str(relation))
                output_csv.append(str(data))
                output_csv.append(str(last_data))
                output_csv.append(str(next_data))
                # 保存为json格式
                # output_json = json.dumps(dict_data)
                # f_output.write(output_json)
                # 保存为csv格式
                csv_write.writerow(output_csv)
        re_num += 1
        print('已处理',re_num,'个关系')

# 关系数据集处理，分为训练、验证和测试集三类
def process_redata(output_re_file):
    f=open(output_re_file,'r',encoding='utf-8')
    f_train=open('./data/re_output_train.csv','a',newline='',encoding='utf-8')
    f_val = open('./data/re_output_val.csv', 'a', newline='', encoding='utf-8')
    f_test = open('./data/re_output_test.csv', 'a', newline='', encoding='utf-8')
    f_reader=csv.reader(f)
    f_train_writer=csv.writer(f_train,dialect='excel')
    f_val_writer = csv.writer(f_val, dialect='excel')
    f_test_writer = csv.writer(f_test, dialect='excel')
    output_train=[]
    output_val=[]
    output_test=[]
    for i,stu in enumerate(f_reader):
        if i>0 and i<=500:
            output_val.append(stu)
        if i>500 and i<=1000:
            output_test.append(stu)
        else:
            output_train.append(stu)
    print(output_train)
    for data_train in count_re(output_train):
        f_train_writer.writerow(data_train)
    for data_val in count_re(output_val):
        f_val_writer.writerow(data_val)
    for data_test in count_re(output_test):
        f_test_writer.writerow(data_test)

# 为人物关系计数
def count_re(output_re):
    output=[]
    num=[1]
    entity_1=output_re[0][0]
    entity_2 = output_re[0][1]
    for data in output_re:
        if data[0]==entity_1 and data[1]==entity_2:
            output.append(num+data)
        else:
            entity_1=data[0]
            entity_2=data[1]
            num[0]=num[0]+1
            output.append(num+data)
    return output

if __name__ == '__main__':
    # 对文件进行分词
    # generate_fenci_file_new(train_file,'./data/news_train_fc_new1.txt')

    # 生成字典
    # generate_vocab_file(seg_train_file,vocab_file)

    # 生成类别
    # generaete_category_dict(train_file,category_file)

    # 匹配关系
    # data_match_re('./data/nlpcc_fc.txt','./data/datasets.json',output_file_csv='./data/re_output.csv')

    # 将文件分开
    process_redata('./data/re_output.csv')







