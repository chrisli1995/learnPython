import random
import numpy as np
from tensorflow.python.client import device_lib
from word_sequence import WordSequence
#定义临界的常量
VOCAB_SIZE_THRESHOLD_CPU=50000

#获取GPU/CPU个数
def _get_avaliable_gpus():
    local_device_protos=device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type=='GPU']

#调用字典是使用cpu还是使用gpu,字典越大选用cpu
def _get_embed_device(vocab_size):
    gpus=_get_avaliable_gpus()
    if not gpus or vocab_size>VOCAB_SIZE_THRESHOLD_CPU:
        return '/cpu:0'
    return '/gpu:0'

# 转换句子
def transform_sentence(sentence,ws,max_len=None,add_end=False):
    encoded=ws.transform(
        sentence,
        max_len=max_len if max_len is not None else len(sentence)
    )
    encoded_len=len(sentence)+(1 if add_end else 0)
    if encoded_len>len(encoded):
        encoded_len=len(encoded)
    return encoded,encoded_len

'''
从数据中随机生成batch_size的数据，然后输出
输入的格式为batch_flow([Q,A],ws,batch_size=32) //Q=[[q1],[q2]] A=[[a1],[a2]]
raw：是否返回原数据
raw=False
返回的是一个生成器next(generator)=q_i_encoded,q_i_len,a_i_encoded,a_i_len
raw=True
返回的是一个生成器next(generator)=q_i_encoded,q_i_len,q_i,a_i_encoded,a_i_len,a_i
add_end为是否添加结束标记
'''
def batch_flow(data,ws,batch_size,raw=False,add_end=True):
    all_data=list(zip(*data))
    # all_data的格式为[([q1],[a1]),([q2],[a2])]
    if isinstance(ws,(list,tuple)):
        assert len(ws)==len(data),'ws的长度必须等于data的长度'
    if isinstance(add_end,bool):
        add_end=[add_end]*len(data)
    else:
        assert (isinstance(add_end,(list,tuple))),'add_end不是boolean，应该是一个list或者一个tuple'
        assert len(add_end)==len(data),'如果add_end是list(tuple),那么add_end的长度应该和输入的数据长度一样'

    mul=2
    if raw:
        mul=3

    while True:
        data_batch=random.sample(all_data,batch_size)
        # data_batch返回的是[([q1],[a1]),([q2],[a2])]
        batches=[[] for i in range(len(data)*mul)]
        max_lens=[]
        for j in range(len(data)):
            max_len=max([
                len(x[j]) if hasattr(x[j],'__len__') else 0
                for x in data_batch
            ])+(1 if add_end[j] else 0)
            max_lens.append(max_len)

        for d in data_batch:
            for j in range(len(data)):
                if isinstance(ws,(list,tuple)):
                    w=ws[j]
                else:
                    w=ws

                #添加结束标记（结尾）
                line=d[j]
                if add_end[j] and isinstance(line,(tuple,list)):
                    line=list(line)+[WordSequence.END_TAG]
                if w is not None:
                    x,x1=transform_sentence(line,w,max_lens[j],add_end[j])
                    batches[j*mul].append(x)
                    batches[j*mul+1].append(x1)
                else:
                    batches[j*mul].append(line)
                    batches[j*mul+1].append(line)
                if raw:
                    batches[j*mul+2].append(line)
        batches=[np.asarray(x) for x in batches]
        # print('xxx',type(batches))
        yield batches

'''
bucket_ind是决定哪个纬度作为切分的依据
n_bucket就是把数据分词多个组（bucket）
其他的与传统的batches操作一样
'''
def batch_flow_bucket(data,ws,batch_size,raw=False,add_end=True,
                      n_bucket=5,bucket_ind=1,debug=False):
    all_data=list(zip(*data))
    # all_data的格式为[([q1],[a1]),([q2],[a2])]
    # set有去重的作用
    lengths=sorted(list(set([len(x[bucket_ind]) for x in all_data])))
    if n_bucket>len(lengths):
        n_bucket=len(lengths)

    # linspace可以创造一个等差数列的numpy，通过这个等差数列的numpy作为lengths索引来完成bucket操作
    splits=np.array(lengths)[
        (np.linspace(0,1,5,endpoint=False)*len(lengths)).astype(int)
    ].tolist()

    splits+=[np.inf]#np.inf表示无限大的正整数

    if debug:
        print(splits)

    # ind_data依据长度将句子进行了分组
    ind_data={}
    for x in all_data:
        l=len(x[bucket_ind])
        for ind,s in enumerate(splits[:-1]):
            # 判断当前的数据的长度在哪个bucket里面
            if l>=s and l<=splits[ind+1]:
                if ind not in ind_data:
                    ind_data[ind]=[]
                ind_data[ind].append(x)
                break

    inds=sorted(list(ind_data.keys()))
    ind_p=[len(ind_data[x])/len(all_data) for x in inds]
    # ind_p为每一个bucket中数据占所有数据的比例
    if debug:
        print(inds)
        print(np.sum(ind_p),ind_p)

    if isinstance(ws,(list,tuple)):
        assert len(ws)==len(data),'len(ws)必须等于len(data)，ws是list或者是tuple'
    if isinstance(add_end,bool):
        add_end=[add_end]*len(data)
    else:
        assert isinstance(add_end,(list,tuple)),'add_end不是Boolean，应该是一个list(tuple) of boolean'
        assert len(add_end)==len(data),'如果add_end是list(tuple)，那么add_end的长度应该和输入的长度是一致的'

    mul=2
    if raw:
        mul=3

    while True:
        choice_ind=np.random.choice(inds,p=ind_p)
        if debug:
            print('choice_ind',choice_ind)
        data_batch=random.sample(ind_data[choice_ind],batch_size)
        batches=[[] for i in range(len(data)*mul)]

        max_lens=[]
        for j in range(len(data)):
            max_len=max([
                len(x[j]) if hasattr(x[j],'__len__') else 0
                for x in data_batch
            ])+(1 if add_end[j] else 0)
            max_lens.append(max_len)

        for d in data_batch:
            for j in range(len(data)):
                if isinstance(ws,(list,tuple)):
                    w=ws[j]
                else:
                    w=ws

                #添加一个结尾
                line=d[j]
                if add_end[j] and isinstance(line,(tuple,list)):
                    line=list(line)+[WordSequence.END_TAG]
                if w is not None:
                    x,xl=transform_sentence(line,w,max_lens[j],add_end[j])
                    batches[j*mul].append(x)
                    batches[j*mul+1].append(xl)
                else:
                    batches[j*mul].append(line)
                    batches[j*mul+1].append(line)
                if raw:
                    batches[j*mul+2].append(line)
        batches=[np.asarray(x) for x in batches]

        yield batches

def test_batch_flow():
    from fake_data import generate
    x_data, y_data, ws_input, ws_target=generate(size=10000)
    flow=batch_flow([x_data,y_data],[ws_input,ws_target],4)
    # print(type(flow))
    x,xl,y,yl=next(flow)
    print(x.shape,y.shape,xl.shape,yl.shape)
    print(x)
    print(xl)
    print(y)
    print(yl)

def test_batch_flow_bucket():
    from fake_data import generate
    x_data, y_data, ws_input, ws_target = generate(size=10000)
    flow = batch_flow_bucket([x_data, y_data], [ws_input, ws_target], 4,debug=True)
    print(flow)
    for _ in range(10):
        x, xl, y, yl = next(flow)
        print(x.shape, y.shape, xl.shape, yl.shape)


if __name__ == '__main__':
    # size=30000
    # print(_get_embed_device(size))
    # test_batch_flow()
    test_batch_flow_bucket()
