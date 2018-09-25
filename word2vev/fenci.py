import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs, sys


def cutwords(froad, targetroad):
    def cut_words(sentence):
        # print sentence
        return " ".join(jieba.cut(sentence)).encode('utf-8')

    # #需要分词的文本
    # f = codecs.open('D:\Python网络爬虫\Word2Vec\jianti1\jianti2.txt', 'r', encoding="utf8")
    f=codecs.open(froad,'r',encoding='utf8')
    target=codecs.open(targetroad,'w',encoding='utf8')
    # #分词以后的结果
    # target = codecs.open("D:\Python网络爬虫\Word2Vec\jianti1\jianti2fc.txt", 'w', encoding="utf8")
    print('open files')
    line_num = 1
    line = f.readline()
    while line:
        print('---- processing ', line_num, ' article----------------')
        line_seg = " ".join(jieba.cut(line))
        target.writelines(line_seg)
        line_num = line_num + 1
        line = f.readline()
    f.close()
    target.close()
    exit()
    # while line:
    #     curr = []
    #     for oneline in line:
    #         # print(oneline)
    #         curr.append(oneline)
    #     after_cut = map(cut_words, curr)
    #     target.writelines(after_cut)
    #     print('saved', line_num, 'articles')
    #     exit()
    #     line = f.readline1()
    # f.close()
    # target.close()

    '''
    import jieba
    import re
    filename='cut_std_zh_wiki_01'
    fileneedCut='std_zh_wiki_01'
    fn=open(fileneedCut,"r",encoding="utf-8")
    f=open(filename,"w+",encoding="utf-8")
    for line in fn.readlines():
        words=jieba.cut(line)
        for w in words:
           f.write(str(w))
    f.close()
    fn.close()
    '''