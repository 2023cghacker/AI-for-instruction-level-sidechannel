"""
    @Author : ling chen
    @Create : 2023/12
    @Last modify: 2023/12
    @Description: 插值方法，从高维数据生成低维数据
"""

import numpy as np
import matplotlib.pyplot as plt
import src.Configuration_matplot
from src.Configuration_01 import DIRECTORY_PREFIX
from src.Model_MainCode.Loadmatfile import loadData, loadTraintest


def Interpolation_down_sampling(X, new_dimension):
    # 计算索引
    indices = np.linspace(0, X.shape[1] - 1, new_dimension)  # 旧索引
    indices = np.trunc(indices).astype(int)  # 取整
    resampled_data = X[:, indices]

    # 绘制前后对比图像
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title('Original Data ')
    plt.plot(X[0])
    plt.subplot(2, 1, 2)
    plt.title('Resampled Data')
    plt.plot(resampled_data[0])
    plt.show()

    return resampled_data


def Interpolation_Up_sampling(X, new_dimension):
    # 计算索引
    old_indices = np.linspace(0, X.shape[1] - 1, X.shape[1])  # 旧索引
    new_indices = np.linspace(0, X.shape[1] - 1, new_dimension)  # 新索引

    # 插值
    upscaled_data = np.zeros((X.shape[0], new_dimension))  # 创建升维后的数据矩阵
    for i in range(X.shape[0]):
        upscaled_data[i] = np.interp(new_indices, old_indices, X[i])

    # 绘制前后对比图像
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title('Original Data')
    plt.plot(X[0])
    plt.subplot(2, 1, 2)
    plt.title('Upscaled Data')
    plt.plot(upscaled_data[0])
    plt.show()

    return upscaled_data


if __name__ == '__main__':
    TraintestName = DIRECTORY_PREFIX + "traintest_(42d100s).mat"
    train_X, train_Y, test_X, test_Y = loadTraintest(TraintestName, name2="train_Y", name4="test_Y")
    # 示例用法
    Interpolation_down_sampling(train_X, new_dimension=30)
    Interpolation_Up_sampling(train_X, new_dimension=500)
