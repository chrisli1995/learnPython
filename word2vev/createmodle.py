import logging
import os.path
import sys
import multiprocessing
from gensim.corpora import wikicorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# if __name__=='__main__':
#     program=os.path.(sbasenameys.argv[0]) #为脚本获取数据集的名称
#     #输出日志
#     logger=logging.getLogger(program)
#     logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
#     logging.root.setLevel(level=logging.INFO)
#     logger.info("running %s"%''.join(sys.argv))
#     #检查输入
#     if len(sys.argv)<4:
#         print(globals()['__doc__']%locals())
#         sys.exit(1)
#     inp,outp1,outp2=sys.argv[1:4]
#     model=Word2Vec(LineSentence(inp),size=400,window=5,min_count=5,workers=multiprocessing.cpu_count())
#     model.save(outp1)
#     model.wv.save_word2vec_format(outp2,binary=False)
def createmord2vec(traindata,modelroad,vectorroad):
    program=os.path.basename('createmord2vec') #为脚本获取数据集的名称
    #输出日志
    logger=logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s"%''.join(sys.argv))
    #训练模型
    modelname=traindata.split('\\')[-1]
    print(modelname)
    modelroad=modelroad+modelname
    print(modelroad)
    vectorroad=vectorroad+modelname
    print(vectorroad)
    model=Word2Vec(LineSentence(traindata),size=400,window=5,min_count=5,workers=multiprocessing.cpu_count())
    model.save(modelroad)
    model.wv.save_word2vec_format(vectorroad,binary=False)