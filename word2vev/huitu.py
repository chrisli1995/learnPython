import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def createdot(embedding,vocabs=0):
    # mpl.rcParams['font.family'] = 'sans-serif'
    # mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    #
    # # x, y = np.loadtxt('test.txt', delimiter=',', unpack=True)
    # x,y=np.load
    # plt.plot(x, y, '*', label='Data', color='black')
    #
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Data')
    # plt.legend()
    # plt.show()


    # demo = [[408.19315, -216.20015], [-479.88858, 24.245243], [-151.56918, 404.6265], [-76.60324, -246.3665],
    #         [315.7837, 277.6758]]
    # x_len = len(demo)
    # y_len = len(demo)
    x = []
    y = []
    for data in embedding:
        x.append(data[0])
        y.append(data[1])
    print(x, y)
    x = np.array(x)
    y = np.array(y)
    # 计算颜色值
    color = np.arctan2(y, x)
    # 绘制散点图
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
    mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.scatter(x, y, s=75, c=color, alpha=0.5)
    for i in range(len(x)):
        plt.annotate(vocabs[i], xy=(x[i], y[i]), xytext=(x[i] + 0.1, y[i] + 0.1))
    # 不显示坐标轴的值
    plt.xticks(())
    plt.yticks(())
    plt.show()
