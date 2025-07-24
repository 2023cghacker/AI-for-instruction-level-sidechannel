import math
from scipy.io import loadmat
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.Configuration_01 import DIRECTORY_PREFIX


def simsilarity(x, y):
    """
    :param x: 向量
    :param y: 向量
    :return: 向量之间的余弦相似度
    """
    if len(x) != len(y):
        print("向量长度不等")
        return
    a = 0
    b = 0
    c = 0
    for i in range(len(x)):
        a = x[i] * y[i] + a
        b = x[i] * x[i] + b
        c = y[i] * y[i] + c
    return (a / (math.sqrt(b) * math.sqrt(c)))


if __name__ == '__main__':
    add_pre = DIRECTORY_PREFIX
    matdata = loadmat(add_pre + '125d_DATA')
    X = matdata['X']
    data = X[1]

    # 余弦相似度，[-1，1]越大越相关
    PI = 3.1415926
    cos_sim_mean = np.zeros((21, 21))  # 计算两个指令集合之间的平均余弦相似度
    cos_sim_var = np.zeros((21, 21))  # 计算两个指令集合之间的余弦相似度方差
    theta = np.zeros((21, 21))
    n = 111 * 110
    cos_sim = np.zeros(n)
    for K in range(21):
        for K2 in range(K, 21):
            print(K, K2)
            t = 0
            for i in range(0 + K * 111, 110 + K * 111):
                for j in range(0 + K2 * 111, 111 + K2 * 111):
                    cos_sim[t] = simsilarity(X[i], X[j])
                    t = t + 1
            cos_sim_mean[K][K2] = np.mean(cos_sim)
            cos_sim_var[K][K2] = np.var(cos_sim)
            # theta[K][K2] = np.arccos(cos_sim_mean[K][K2]) / PI * 180
            print("余弦相似度均值：", cos_sim_mean[K][K2], "余弦相似度方差：", cos_sim_var[K][K2])

    # 绘图
    ax = plt.subplots()  # 调整画布大小
    ax = sns.heatmap(cos_sim_mean)  # 画热力图
    plt.show()
