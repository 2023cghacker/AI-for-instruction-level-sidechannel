# -*- coding:UTF-8 -*-#
"""
    @filename:filter.py
    @Author:chenling
    @Create:2023/12/25
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, filtfilt

from src.Configuration_02 import DIRECTORY_PREFIX
from src.Model_MainCode.Loadmatfile import loadData, extract_name


def low_pass_filter_1d(signal, cutoff_freq, fs):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist

    # 设计一个低通Butterworth滤波器
    b, a = butter(N=6, Wn=normal_cutoff, btype='low', analog=False, output='ba')

    # 应用滤波器
    filtered_signal = filtfilt(b, a, signal)

    # 绘制原始信号和滤波后的信号
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal, label='Original Signal')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(filtered_signal, label='Filtered Signal (Low Pass)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    return filtered_signal


if __name__ == '__main__':
    dataName = DIRECTORY_PREFIX + "DATA_(GAN500d100s).mat"
    X, Y = loadData(dataName, )
    # 示例用法
    # 调用函数进行低通滤波
    cutoff_frequency = 10  # 截止频率
    sampling_frequency = 100  # 采样频率
    filterX=low_pass_filter_1d(X, cutoff_frequency, sampling_frequency)


    scipy.io.savemat(dataName, {'X': filterX, 'Y': Y})
    print("训练集测试集已经保存在：", extract_name(dataName), "\n")

