import re
import codecs
import sys
import pickle
from tqdm import tqdm
from word_sequence import WordSequence

# 就是打个逗号
def make_split(line):
    if re.match(r'.*([，…?!\.,!？])$', ''.join(line)):
        return []
    return [',']

# 判断是否为好的句子
def good_line(line):
    if len(re.findall(r'[a-zA-Z0-9]',''.join(line)))>2:
        return False
    return True

#通过正则修改一些文件
def regular(sen):
    sen=re.sub(r'\.{3,100}','…',sen)
    sen=re.sub(r'…{2,100}','…',sen)
    sen=re.sub(r'[\,]{1,100}','，',sen)
    sen=re.sub(r'[\.]{1,100}','。',sen)
    sen=re.sub(r'[\?]{1,100}','？',sen)
    sen=re.sub(r'[!]{1,100}','！',sen)
    return sen

def main(limit=20,x_limit=3,y_limit=6):
    print('extract lines')
    fp=codecs.open('.//data//dgk_shooter_min.conv','r',errors='ignore',encoding='utf-8')
    groups=[]
    group=[]

    for line in tqdm(fp):
        if line.startswith('M'):
            line=line.replace('\n','')
            if '/' in line:
                line=line[2:].split('/')
            else:
                line=list(line[2:])
            line=line[:-1]
            group.append(list(regular(''.join(line))))
        else:
            if group:
                groups.append(group)
                group=[]

    if group:
        groups.append(group)
        group=[]
    print('extract group')
    # print(groups[0])
    #以上返回的数据的格式为[[[第一行]，[第二行],[第三行]]，[[..],[..]]],
    #groups[0]的数据为[['畹', '华', '吾', '侄'], ['你', '接', '到', '这', '封', '信', '的', '时', '候'], ['不', '知', '道', '大', '伯', '还', '在', '不', '在', '人', '世', '了']]

    x_data=[]
    y_data=[]
    for group in tqdm(groups):
        for i,line in enumerate(group):
            last_line=None
            if i>0:
                last_line=group[i-1]
                if not good_line(last_line):
                    last_line=None
            next_line=None
            if i<len(group)-1:
                next_line=group[i+1]
                if not good_line(next_line):
                    next_line=None
            next_next_line=None
            if i<len(group)-2:
                next_next_line=group[i+2]
                if not good_line(next_next_line):
                    next_next_line=None

            if next_line:
                x_data.append(line)
                y_data.append(next_line)
            if last_line and next_line:
                x_data.append(last_line+make_split(last_line)+line)
                y_data.append(next_line)
            if next_line and next_next_line:
                x_data.append(line)
                y_data.append(next_line+make_split(next_line)+next_next_line)

    print(len(x_data),len(y_data))

    for ask,answer in zip(x_data[:20],y_data[:20]):
        print(''.join(ask))
        print(''.join(answer))
        print('-'*20)

    data=list(zip(x_data,y_data))
    data=[
        (x,y)
        for x,y in data
        if len(x)<limit
        and len(y)<limit
        and len(y)>=y_limit
        and len(x)>=x_limit
    ]
    x_data,y_data=zip(*data)# zip()与zip(*)功能相反，返回的是同样是一个元组的列表[(x_data),(y_data)]
    print('xdata',x_data)
    print('ydata',y_data)

    print('fit word_sequence-构建词表')
    ws_input=WordSequence()
    ws_input.fit(x_data+y_data)

    print('dump-保存数据')
    pickle.dump(
        (x_data,y_data),
        codecs.open('.//data//chatbot.pkl','wb')
    )
    pickle.dump(ws_input,codecs.open('.//data//ws.pkl','wb'))
    print('done-完成词表和数据集的构建')


if __name__ == '__main__':
    main()

