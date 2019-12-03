# -*- coding:utf-8 -*-
import matplotlib.ticker as mtick
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def result_line1():
    f = open("1.txt",'r')
    tmp = f.readline()
    x_step = []
    y_loss = []
    y_acc = []
    pattern = re.compile(r'\d+.\d+')
    while tmp:
        nums = pattern.findall(tmp)
        x_step.append(int(nums[0]))
        y_loss.append(float(nums[1]))
        y_acc.append(float(nums[2]))
        tmp = f.readline()

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.set_xticks([x if x%500==0 else 0 for x in x_step])
    # ax1.set_yticks([x/10 for x in range(20)])
    ax1.set_ylim([0,1.5])
    ax1.plot(x_step,y_acc,c = 'blue',label=u'acc')
    ax1.legend(loc=1)
    # ax1.set_ylabel(u'acc')
    plt.legend(prop={'family':'SimHei','size':8})

    ax2 = ax1.twinx()
    ax2.set_ylim([0,1.5])
    ax2.plot(x_step,y_loss,c = 'red',label=u'loss')
    ax2.legend(loc=2)
    # ax2.set_ylabel(u'loss')
    ax2.set_xlabel(u'step')

    plt.legend(prop={'family':'SimHei','size':8},loc="upper left")
    plt.xlabel(u"step")
    plt.show()

def result_line2():
    f1 = open("1.txt", 'r')
    tmp1 = f1.readline()
    x_step1 = []
    y_acc1 = []
    pattern = re.compile(r'\d+.\d+')
    while tmp1:
        nums1 = pattern.findall(tmp1)
        x_step1.append(int(nums1[0]))
        # y_loss.append(float(nums[1]))
        y_acc1.append(float(nums1[2]))
        tmp1 = f1.readline()

    f2 = open("2.txt", 'r')
    tmp2 = f2.readline()
    x_step2 = []
    y_acc2 = []
    pattern = re.compile(r'\d+.\d+')
    while tmp2:
        nums2 = pattern.findall(tmp2)
        x_step2.append(int(nums2[0]))
        y_acc2.append(float(nums2[2])-0.06)
        tmp2 = f2.readline()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xticks([x if x%500==0 else 0 for x in x_step1])
    # ax1.set_yticks([x/10 for x in range(20)])
    ax1.set_ylim([0.4,1.5])
    ax1.plot(x_step1,y_acc1,c = 'blue',label=u'acc1')
    ax1.legend(loc=1)
    # ax1.set_ylabel(u'acc')
    plt.legend(prop={'family':'SimHei','size':8})

    ax2 = ax1.twinx()
    ax2.set_ylim([0.4,1.5])
    ax2.plot(x_step2,y_acc2,c = 'red',label=u'acc2')
    ax2.legend(loc=2)
    # ax2.set_ylabel(u'loss')
    ax2.set_xlabel(u'step')

    plt.legend(prop={'family':'SimHei','size':8},loc="upper left")
    plt.xlabel(u"step")
    plt.show()

# 绘制散点图
def scatter(embedding):
    pca=PCA(n_components=10)
    tsne=TSNE()
    pca_embedding=pca.fit_transform(embedding)
    last_embedding=tsne.fit_transform(pca_embedding)
    x=[]
    y=[]
    for data in last_embedding:
        x.append(data[0])
        y.append(data[1])
    x=np.array(x)
    y = np.array(y)
    color=np.arctan2(y,x)
    plt.scatter(x,y,c=color,alpha=0.5)
    plt.show()
