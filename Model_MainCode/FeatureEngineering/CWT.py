import numpy as np
import pywt
from scipy.io import loadmat
from src.Configuration_03 import DIRECTORY_PREFIX
from src import Configuration_matplot
import matplotlib.pyplot as plt


def normalize_array(array):
    # 计算数组的最大值和最小值
    max_value = np.max(array)
    min_value = np.min(array)

    # 将数组归一化
    normalized_array = (array - min_value) / (max_value - min_value)
    return normalized_array


def CWT(data, totalscal):
    """
    :param data: 功耗迹，一维时序数据
    :totalscal: 是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
    :return: 二维时频域数据
    """
    # wavename = "cgau8"  # 小波函数
    # fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
    # cparam = 2 * fc * totalscal  # 常数c
    # scales = cparam / np.arange(totalscal+1, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
    # cwtmatr, frequencies = pywt.cwt(data, scales, wavename)  # 连续小波变换模块


    wavename = "cmor"  # 小波函数
    scales = np.linspace(1, len(data), totalscal)
    cwtmatr, frequencies = pywt.cwt(data, scales, wavename)


    return [cwtmatr, frequencies]


def plot_TFdomain(data, frequencies, cwtdata):
    """
    :param data: 功耗迹数据（单个）
    :param frequencies: 频率分量
    :param cwtdata: CWT后二维时频域数据（单个）
    :return: 绘图
    """
    t = np.linspace(0, len(data), len(data))
    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.plot(t, data)
    plt.xlabel(u"time(s)")
    plt.title(u"Power trace")

    plt.subplot(212)
    plt.contourf(t, frequencies, abs(cwtdata))
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.title(u"t-f-domain")
    plt.subplots_adjust(hspace=0.6)  # 调整边距和子图的间距 hspace为子图之间的空间保留的高度，平均轴高度的一部分
    plt.show()

# address = DIRECTORY_PREFIX + '1T&2T_DATA_l(500d100s).mat'
# matdata = loadmat(address)  # 读取Mat文件, 文件里有训练集和测试集
#
# X = matdata['X_1']
# [cwtmatr, frequencies] = CWT(X[0])
#
# plot_TFdomain(X[0], frequencies, cwtmatr)
# cwtmatr = normalize_array(cwtmatr)
# plot_TFdomain(X[0], frequencies, cwtmatr)
