import random
import numpy as np
from word_sequence import WordSequence

def generate(max_len=10,size=1000,same_len=False,seed=0):
    dictionary={
        'a':'1',
        'b': '2',
        'c': '3',
        'd': '4',
        'aa': '1',
        'bb': '2',
        'cc': '3',
        'dd': '4',
        'aaa': '1',
    }

    if seed is not None:
        random.seed(seed)

    input_list=sorted(list(dictionary.keys()))

    x_data=[]
    y_data=[]

    for x in range(size):
        a_len=int(random.random()*max_len)+1
        x=[]
        y=[]
        for _ in range(a_len):
            word=input_list[int(random.random()*len(input_list))]
            x.append(word)
            y.append(dictionary[word])
            if not same_len:
                if y[-1]=='2':
                    y.append('2')
                elif y[-1]=='3':
                    y.append('3')
                    y.append('4')
        x_data.append(x)
        y_data.append(y)

    ws_input=WordSequence()
    ws_input.fit(x_data)

    ws_target=WordSequence()
    ws_target.fit(y_data)

    return x_data,y_data,ws_input,ws_target

def test():
    x_data, y_data, ws_input, ws_target=generate()
    all_data=list(zip(*[x_data,y_data]))
    data_batch = random.sample(all_data, 4)
    max_lens = []
    for j in range(len([x_data,y_data])):
        max_len = max([
            len(x[j]) if hasattr(x[j], '__len__') else 0
            for x in data_batch
        ]) + 1
        max_lens.append(max_len)
    # print(max_lens)
    # print(data_batch)
    # print(len(data_batch[0]))
    print('x_data',x_data)
    print('y_data', y_data)
    # print('ws_input', ws_input)
    # print('ws_target', ws_target)
    # assert len(x_data)==1000
    # print(len(y_data))
    # assert len(y_data) == 1000
    # print(np.max([len(x) for x in x_data]))
    # assert np.max([len(x) for x in x_data])==10
    # print(len(ws_input))
    # assert len(ws_input)==14
    # print(len(ws_target))
    print(all_data)
    # print([ws_input,ws_target])
    # print(ws_target.dict)
    # print(ws_input.dict)

if __name__ == '__main__':
    test()

