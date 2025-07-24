import sys
import scipy
import numpy as np
from collections import Counter
import math
import time

from matplotlib import pyplot as plt

from src.Model_MainCode.FeatureEngineering.plot_gain_ratios import plot_gain_ratios


def My_featureAlgorithm(train_X, train_Y_i, train_Y_d, test_X, K):
    """
    X：功耗数据集
    K：选择多少个特征点（作为输出）
    """

    # 判断数据维度
    if len(train_X.shape) == 3:
        # 处理二维时频域数据
        num_features = train_X.shape[1] * train_X.shape[2]
        data_shape = train_X.shape[1:]
        train_X = train_X.reshape((train_X.shape[0], num_features))
        test_X = test_X.reshape((test_X.shape[0], num_features))
    else:
        # 处理一维时序数据
        num_features = train_X.shape[1]
        data_shape = None

    # 计算信息增益比值
    gain_ratios = np.zeros(num_features)
    calculate_information_gain_ratios(train_X, train_Y_i, train_Y_d, gain_ratios)
    top_K_indices = np.argpartition(gain_ratios, -K)[-K:]

    # scipy.io.savemat('feature_'+str(K)+'.mat', {'gain_ratios': gain_ratios, 'top_K_indices': top_K_indices, 'data_shape': data_shape})

    # 绘制信息增益比值图
    # plot_gain_ratios(gain_ratios, top_K_indices, data_shape)

    # 选择前K个特征点
    selected_train_X = train_X[:, top_K_indices]
    selected_test_X = test_X[:, top_K_indices]

    return selected_train_X, selected_test_X


def entropy(column):
    """
    计算熵，如果unique_values.size大于100，分成30个小区间
    """
    unique_values = np.unique(column)
    if unique_values.size <= 100:
        counter = Counter(column)
        total_count = len(column)
        entropy_val = 0.0

        for count in counter.values():
            probability = count / total_count
            entropy_val -= probability * math.log2(probability)
    else:
        min_value, max_value = np.min(column), np.max(column)
        bins = np.linspace(min_value, max_value, 31)
        digitized = np.digitize(column, bins) - 1

        entropy_val = 0.0
        total_count = len(column)

        for i in range(len(bins) - 1):
            bin_count = np.sum(digitized == i)
            if bin_count > 0:
                probability = bin_count / total_count
                entropy_val -= probability * math.log2(probability)

    return entropy_val


def conditional_entropy(data, labels):
    """
    计算条件熵，
    """
    unique_values = np.unique(data)
    total_count = len(data)
    cond_entropy = 0.0

    if unique_values.size <= 100:
        for value in unique_values:
            subset = labels[data == value]
            prob = len(subset) / total_count
            cond_entropy += prob * entropy(subset)
    else:
        min_value, max_value = np.min(data), np.max(data)
        bins = np.linspace(min_value, max_value, 31)
        digitized = np.digitize(data, bins) - 1

        for i in range(len(bins) - 1):
            subset = labels[digitized == i]
            if len(subset) > 0:
                prob = len(subset) / total_count
                cond_entropy += prob * entropy(subset)

    return cond_entropy


def information_gain(data, labels):
    """
    计算信息增益
    """
    total_entropy = entropy(labels)
    cond_entropy = conditional_entropy(data, labels)
    return total_entropy - cond_entropy


def progress_bar(nowindex, totalindex):
    progress = nowindex / totalindex
    length = 30
    block = int(round(length * progress))
    progress_text = '█' * block + '-' * (length - block)
    sys.stdout.write(f'\r信息增益计算进度条[{progress_text}]{nowindex}\{totalindex} {progress * 100:.2f}%')
    sys.stdout.flush()


def calculate_information_gain_ratios(data, labels_A, labels_B, gain_ratios):
    """
    计算每个特征点对标签A和标签B的信息增益比值
    """
    num_samples, num_features = data.shape
    #     gain_ratios = []
    for feature_idx in range(num_features):
        feature_values = data[:, feature_idx]
        gain_A = information_gain(feature_values, labels_A)
        gain_B = information_gain(feature_values, labels_B)
        ratio = gain_A / gain_B if gain_B != 0 else 0
        gain_ratios[feature_idx] = ratio

        # 绘制进度条
        progress_bar(feature_idx+1, num_features)

#     return np.array(gain_ratios)
