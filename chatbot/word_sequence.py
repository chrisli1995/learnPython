import numpy as np

class WordSequence(object):
    PAD_TAG='<pad>'
    UNK_TAG='<unk>'
    START_TAG='<s>'
    END_TAG='</s>'

    PAD=0
    UNK=1
    START=2
    END=3

    def __init__(self):
        self.dict={
            WordSequence.PAD_TAG:WordSequence.PAD,
            WordSequence.UNK_TAG:WordSequence.UNK,
            WordSequence.START_TAG:WordSequence.START,
            WordSequence.END_TAG:WordSequence.END,
        }
        self.fited=False

    #把字转换为index
    def to_index(self,word):
        assert self.fited,'WordSequence尚未进行fit操作'
        if word in self.dict:
            return self.dict[word]
        return WordSequence.UNK

    #把index转换为字
    def to_word(self,index):
        assert self.fited, 'WordSequence尚未进行fit操作'
        for k,v in self.dict.items():
            if v==index:
                return k
        return WordSequence.UNK

    #定义大小
    def size(self):
        assert self.fited, 'WordSequence尚未进行fit操作'
        return len(self.dict)+1

    def __len__(self):
        return self.size()

    # 定义字典
    def fit(self,sentences,min_count=5,max_count=None,max_feature=None):
        assert not self.fited,'WordSequence只能fit一次'
        count={}
        for sentence in sentences:
            arr=list(sentence)
            for a in arr:
                if a not in count:
                    count[a]=0
                count[a]+=1
        if min_count is not None:
            count={k:v for k,v in count.items() if v>=min_count}

        if max_count is not None:
            count = {k: v for k, v in count.items() if v <= max_count}
        # print('xxx',count)
        #dict用于将为每个单词分配一个index
        self.dict={
            WordSequence.PAD_TAG : WordSequence.PAD,
            WordSequence.UNK_TAG : WordSequence.UNK,
            WordSequence.START_TAG : WordSequence.START,
            WordSequence.END_TAG : WordSequence.END,
        }
        if isinstance(max_feature,int):
            count=sorted(list(count.items()),key=lambda x:x[1])
            if max_feature is not None and len(count)>max_feature:
                count=count[-int(max_feature):]
            for w,_ in count:
                self.dict[w]=len(self.dict)
        else:
            for w in sorted(count.keys()):
                self.dict[w]=len(self.dict)

        self.fited=True

    #句子->向量
    def transform(self,sentence,max_len=None):
        assert self.fited,'WordSequence只能fit一次'
        if max_len is not None:
            r=[self.PAD]*max_len
        else:
            r=[self.PAD]*len(sentence)

        for index,a in enumerate(sentence):
            if max_len is not None and index>=len(r):
                break
            r[index]=self.to_index(a)
        return np.array(r)

    #向量->句子
    def inverse_transform(self,indices,
                          ignore_pad=False,ignore_unk=False,
                          ignore_start=False,ignore_end=False):
        ret=[]
        for i in indices:
            word=self.to_word(i)
            if word==WordSequence.PAD_TAG and ignore_pad:
                continue
            if word==WordSequence.UNK_TAG and ignore_unk:
                continue
            if word==WordSequence.START_TAG and ignore_start:
                continue
            if word==WordSequence.END_TAG and ignore_end:
                continue
            ret.append(word)
        return ret

def test():
    ws=WordSequence()
    print(type(ws))
    ws.fit([
        ['你','好','啊'],
        ['你','好','哦']
    ])
    print(ws.dict)

    indice=ws.transform(['你','们'])
    print(indice)

    back=ws.inverse_transform(indice)
    print(back)
    print('wslength:',len(ws))

if __name__ == '__main__':
    test()