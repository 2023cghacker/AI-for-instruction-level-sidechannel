import scipy
from scipy.io import loadmat
import numpy as np
from src.Configuration_03 import DIRECTORY_PREFIX
from src.Model_MainCode.FeatureEngineering.CWT import CWT, plot_TFdomain
import matplotlib.pyplot as plt


def calculate_mean_variance_label(arr, labels):
    """
    :param arr: 输入
    :param labels: 标签
    :return: 方差和均值
    """
    num_samples = arr.shape[0]
    num_positions = arr.shape[1]
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)

    # 计算每个位置上的均值和方差
    mean_arr_all = np.mean(arr, axis=0)
    variance_arr_all = np.var(arr, axis=0)

    # 创建存储结果的数组
    sample_count_per_label = np.zeros(num_labels)
    mean_arr_label = np.zeros((num_labels, num_positions))
    variance_arr_label = np.zeros((num_labels, num_positions))

    for i, label in enumerate(unique_labels):
        label_samples = arr[labels == label]
        num_label_samples = len(label_samples)

        sample_count_per_label[i] = num_label_samples
        mean_arr_label[i] = np.mean(label_samples, axis=0)
        variance_arr_label[i] = np.var(label_samples, axis=0)

    return mean_arr_all, variance_arr_all, sample_count_per_label, mean_arr_label, variance_arr_label



if __name__ == '__main__':
    DataName = DIRECTORY_PREFIX + "1T&2T_DATA_m(500d100s).mat"  # 数据集文件名称
    NewDataName = DIRECTORY_PREFIX + "1Tl(cwt)DATA_m(100s).mat"  # 时频域转化后的数据集文件名称
    matdata = loadmat(DataName)  # 读取Mat文件
    X = matdata['X_1']
    Y = matdata['Y_1'][0]
    print("数据集地址；", DataName)
    print("数据集输入规模；", X.shape, "数据集标签规模", Y.shape)

    mean_arr_all, variance_arr_all, sample_count_per_label, mean_arr_label, variance_arr_label = calculate_mean_variance_label(
        X, Y)
    print(mean_arr_label.shape, variance_arr_label.shape)

    var_sum = np.sum(variance_arr_label, axis=0)
    t = variance_arr_all / var_sum
    indices = np.argsort(t)[-30:]

    plt.subplot(3, 1, 1)
    plt.plot(variance_arr_all)
    plt.subplot(3, 1, 2)
    plt.plot(var_sum)
    plt.subplot(3, 1, 3)
    plt.plot(t)
    plt.show()
