import numpy as np
import pywt
from scipy.io import loadmat
from src.Configuration_04 import DIRECTORY_PREFIX
from src import Configuration_matplot
import matplotlib.pyplot as plt

from src.Model_MainCode.Loadmatfile import loadData
import numpy as np


def GADF(data):
    """
    计算格拉米角场（GADF）
    参数：
    data：输入序列数据（一维数组）
    返回：
    gadf：格拉米角场（二维数组）
    """
    N = len(data)
    data_scaled = (data - np.min(data)) / (np.max(data) - np.min(data))  # 将数据缩放到 [0, 1] 范围内
    angles = np.arccos(data_scaled)  # 计算角度
    gadf = np.zeros((N - 1, N - 1))  # 初始化二维数组
    for i in range(N - 1):
        for j in range(N - 1):
            gadf[i, j] = angles[i + 1] - angles[j]
    return gadf


if __name__ == '__main__':
    X, Y = loadData(DIRECTORY_PREFIX + "DATA_m(750d200s).mat")

    # [cwtmatr, t, frequencies] = WVD(X[0,:], fs=50)
    for i in range(1, 1000, 100):
        cwtmatr = GADF(X[i, :])
        t = np.linspace(0, len(cwtmatr), len(cwtmatr))
        f = np.linspace(0, len(cwtmatr), len(cwtmatr))
        plt.contourf(t, f, cwtmatr)
        plt.title(i)
        plt.pause(0.01)
    # print(cwtmatr.shape)
    # plot_TFdomain(X[0, :], frequencies, cwtmatr)
    # cwtmatr = normalize_array(cwtmatr)
    # plot_TFdomain(X[0], frequencies, cwtmatr)
