import fenci
import delete
import createmodle
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
from sklearn.manifold import TSNE
import huitu


if __name__ == '__main__':
    # #中文分词
    # fenci.cutwords('D:\\nstart\\NLP\\word2vev\\data\\novel\\斗破苍穹.txt',
    #                'D:\\nstart\\NLP\\word2vev\data\\novel\\斗破苍穹fctest.txt')

    # #删除特殊符号
    # delete.filte('D:\\nstart\\NLP\\word2vev\data\\novel\\斗破苍穹fc.txt');

    # #创建模型
    # createmodle.createmord2vec('D:\\nstart\\NLP\\word2vev\data\\novel\\斗破苍穹fctest.txt',
    #                            'D:\\nstart\\NLP\\word2vev\\model',
    #                            'D:\\nstart\\NLP\\word2vev\\vector')

    # #测试模型
    # model=Word2Vec.load('D:\\nstart\\NLP\\word2vev\\model\\model斗破苍穹fctest.txt')
    # res=model.most_similar('薰儿')
    # print(res)

    vector=KeyedVectors.load_word2vec_format('D:\\nstart\\NLP\\word2vev\\vector\\vector斗破苍穹fctest.txt')
    # #将所有词汇放入列表
    # words=[]
    # for word in vector.vocab:
    #     words.append(word)
    # print('单词的总数为',len(words))
    # print('每个词的维度为',len(vector[words[0]]))
    # print('单词',words[0],'的向量表示为',vector[words[0]])
    # for similar_word in vector.similar_by_word('萧炎',topn=5):
    #     print(similar_word[0],similar_word[1])
    # for result_word in vector.most_similar(positive=['男人','女人'],negative='国王',topn=5):
    #     print('word:',result_word[0],'similarity:',result_word[1])

    renwu=['萧炎','萧薰儿','小医仙','云韵','美杜莎']
    vocabs=[]
    embedding=np.array([])
    for word in renwu:
        if word in vector.vocab and not renwu in vocabs:
            vocabs.append(word)
            embedding=np.append(embedding,vector[word])
    print(len(vocabs))
    embedding=embedding.reshape(len(vocabs),400)
    tsne=TSNE()
    low_embedding=tsne.fit_transform(embedding)
    print(low_embedding)
    huitu.createdot(low_embedding,vocabs)




