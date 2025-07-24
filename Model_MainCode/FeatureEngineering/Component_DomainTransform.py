import sys
import time
import pywt
import scipy
from scipy.io import loadmat
import numpy as np
from src.Configuration_04 import DIRECTORY_PREFIX
from src.Model_MainCode.FeatureEngineering.CWT import CWT, plot_TFdomain
from src.Model_MainCode.FeatureEngineering.GADFs import GADF
from src.Model_MainCode.Loadmatfile import loadData
from src import Configuration_matplot
import matplotlib.pyplot as plt


def Component_DomainTransform(Data, freqlen):
    """
    :param Data: 二维数组，每个样本是一维
    :param freqlen: 转换后的频域长度
    :return: 三维数组，每个样本是二维
    """
    print("\nbefore DomainTransform-Component:", Data.shape)
    tf_data = np.zeros((len(Data), len(Data[0]), freqlen))
    # scales = np.linspace(1, freqlen, freqlen)
    scales = np.logspace(np.log10(1), np.log10(100), num=50)  # 对数间隔尺度

    total_size = len(Data)  # 总大小
    for i in range(total_size):
        [cwtmatr, frequencies] = CWT(Data[i], scales)
        cwtmatr_real = np.real(cwtmatr)  # 仅保留实部
        cwtmatr_real = np.transpose(cwtmatr_real)
        tf_data[i] = cwtmatr_real

        # 绘制进度条
        progress = i / total_size
        progress_bar(progress, total_size)

    print("\nafter DomainTransform-Component:", tf_data.shape)
    # plot_TFdomain2(Data[0, :], frequencies, cwtmatr)
    return tf_data


# 进度条函数
def progress_bar(progress, total_size):
    length = 30
    block = int(round(length * progress))
    progress_text = '█' * block + '-' * (length - block)
    sys.stdout.write(f'\rDomainTransform-Component runing:[{progress_text}]{total_size} {progress * 100:.2f}%')
    sys.stdout.flush()


def CWT(data, scales):
    """
    :param data: 功耗迹，一维时序数据
    :totalscal: 是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
    :return: 二维时频域数据
    """

    wavename = "cmor1.5-1.0"  # 小波函数
    cwtmatr, frequencies = pywt.cwt(data, scales, wavename)

    return [cwtmatr, frequencies]


def GADFTransform(Data):
    """
    :param Data: 二维数组，每个样本是一维
    :return: 三维数组，每个样本是二维
    """
    print("\nbefore DomainTransform-Component:", Data.shape)
    output_data = np.zeros((len(Data), len(Data[0]) - 1, len(Data[0]) - 1))

    # 进度更新
    total_size = len(Data)  # 总大小

    for i in range(len(Data)):
        cwtmatr = GADF(Data[i, :])
        output_data[i] = cwtmatr

        # 绘制进度条
        progress = i / total_size
        progress_bar(progress, total_size)

    print("\nafter DomainTransform-Component:", output_data.shape)
    return output_data


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

def plot_TFdomain2(data, frequencies, cwtdata):
    """
    :param data: 功耗迹数据（单个）
    :param frequencies: 频率分量
    :param cwtdata: CWT后二维时频域数据（单个）
    :return: 绘图
    """
    t = np.linspace(0, len(data), len(data))
    plt.figure(figsize=(10, 10))
    plt.contourf(t, frequencies, abs(cwtdata))
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.title(u"t-f-domain")
    plt.subplots_adjust(hspace=0.6)  # 调整边距和子图的间距 hspace为子图之间的空间保留的高度，平均轴高度的一部分
    plt.show()

# if __name__ == '__main__':
#     DataName = DIRECTORY_PREFIX + "DATA_m(750d200s).mat"
#     X, Y = loadData(DIRECTORY_PREFIX + "DATA_m(750d200s).mat")
#     print("数据集地址；", DataName)
#     print("数据集输入规模；", X.shape, "数据集标签规模", Y.shape)
#
#     new_X = CWTTransform(X)
#
#     NewDataName = DIRECTORY_PREFIX + "1T_DATA_m(cwt100s).mat"  # 时频域转化后的数据集文件名称
#     scipy.io.savemat(NewDataName, {'X': new_X, 'Y': Y})
