import math
import time
from collections import Counter, defaultdict
import scipy
from scipy.io import loadmat
import numpy as np
from src.Configuration_04 import DIRECTORY_PREFIX
import matplotlib.pyplot as plt
import src.Configuration_matplot
from src.Model_MainCode.Loadmatfile import loadData


def entropy(column):
    counter = Counter(column)
    total_count = len(column)
    entropy_val = 0.0

    for count in counter.values():
        probability = count / total_count
        entropy_val -= probability * math.log2(probability)

    return entropy_val


def calculate_entropy(data):
    """
    :param data: n*m维的数据，n是样本数，每个样本有m个特征点
    :return: 返回每个特征点上的信息熵
    """
    num_samples, num_dimensions = data.shape
    entropies = []

    for dim in range(num_dimensions):
        column_values = [data[row][dim] for row in range(num_samples)]
        entropies.append(entropy(column_values))

    return np.array(entropies)


def calculate_entropy_3d(data):
    """
       :param data: n*m*f维的数据，n是样本数，每个样本有m*f个特征点
       :return: 返回每个特征点上的信息熵
       """
    num_samples, num_m, num_f = data.shape
    data_reshaped = data.reshape(num_samples, num_m * num_f)

    entropies = []
    for dim in range(num_m * num_f):
        column_values = [data_reshaped[row][dim] for row in range(num_samples)]
        entropies.append(entropy(column_values))

    entropies_reshaped = np.array(entropies).reshape(num_m, num_f)

    return entropies_reshaped


def calculate_group_entropies(data, labels):
    """
    :param data: n*m的数据，n是样本数，每个样本有m个特征点
    :param labels: 每个样本数据对应的标签
    :return: group_entropies返回每个标签下的每个特征点的信息熵，(eg：有 5类标签，则返回的是5*m的数组)

    """
    num_samples, num_dimensions = data.shape
    unique_labels = np.unique(labels)
    label_groups = defaultdict(list)

    for i in range(num_samples):
        label_groups[labels[i]].append(data[i])

    group_entropies = []

    for label in unique_labels:
        group_data = np.array(label_groups[label])
        entropies = []

        for dim in range(num_dimensions):
            column_values = group_data[:, dim]
            entropies.append(entropy(column_values))

        group_entropies.append(entropies)

    return np.array(group_entropies)


def calculate_group_entropies_3d(data, labels):
    """
    :param data: n*m*f的数据，n是样本数，每个样本有m*f个特征点
    :param labels: 每个样本数据对应的标签
    :return: group_entropies返回每个标签下的每个特征点的信息熵，(eg：有 5类标签，则返回的是5个m*f的数组)

    """
    num_samples, num_m, num_f = data.shape
    unique_labels = np.unique(labels)
    label_groups = defaultdict(list)

    for i in range(num_samples):
        label_groups[labels[i]].append(data[i])

    group_entropies = []

    for label in unique_labels:
        group_data = np.array(label_groups[label])
        group_data_reshaped = group_data.reshape(-1, num_m * num_f)
        entropies = []

        for dim in range(num_m * num_f):
            column_values = group_data_reshaped[:, dim]
            entropies.append(entropy(column_values))

        entropies_reshaped = np.array(entropies).reshape(num_m, num_f)
        group_entropies.append(entropies_reshaped)

    return np.array(group_entropies)


# def Component_FeatureSelect(train_X, train_Y, test_X, test_Y):
#     n, m = train_X.shape
#     print(n, m)
#     device_label = np.zeros(n)
#     for i in range(int(n / 1000)):
#         device_label[i * 1000:(i + 1) * 1000] = i
#     print(device_label)
#     total_entropy = calculate_entropy(train_X)
#     class_entropy = calculate_group_entropies(train_X, train_Y)
#     device_entropy = calculate_group_entropies(train_X, device_label)
#     # total_entropy = calculate_entropy_3d(train_X)
#     # class_entropy = calculate_group_entropies_3d(train_X, train_Y)
#     # device_entropy = calculate_group_entropies_3d(train_X, device_label)
#     # print("信息熵", total_entropy.shape, class_entropy.shape, device_entropy.shape)
#     class_ratio = np.sum(total_entropy / class_entropy, axis=0)
#     device_ratio = np.sum(total_entropy / device_entropy, axis=0)
#     ratio = class_ratio / device_ratio
#     print("熵比", class_ratio.shape, device_ratio.shape, ratio.shape)
#
#     k = 300
#     indices1 = np.argpartition(-class_ratio, k)[:k]  # 选取最大的点
#     poi1 = class_ratio[indices1]
#     indices2 = np.argpartition(device_ratio, k)[:k]  # 选取最小的点
#     poi2 = device_ratio[indices2]
#     indices3 = np.argpartition(-ratio, k)[:k]  # 选取最小的点
#     poi3 = device_ratio[indices3]
#     '''
#     绘图
#     '''
#     # t = np.linspace(0, len(train_X[0]), len(train_X[0]))
#     # f = np.linspace(0, len(train_X[0][0]), len(train_X[0][0]))
#     plt.figure(figsize=(15, 8))
#     plt.subplot(2, 2, 1)
#     plt.title("功耗迹")
#     # plt.contourf(t, f, train_X[0].T)
#     plt.plot(train_X[0, :])
#     plt.subplot(2, 2, 2)
#     plt.plot(total_entropy)
#     # plt.contourf(t, f, total_entropy.T)
#     plt.title("总信息熵")
#     plt.subplot(2, 2, 3)
#     plt.plot(class_ratio)
#     # plt.contourf(t, f, class_ratio.T)
#     plt.scatter(indices1, poi1, color="red")
#     plt.title("类间熵比和峰值点")
#     plt.subplot(2, 2, 4)
#     plt.plot(device_ratio)
#     # plt.contourf(t, f, device_ratio.T)
#     plt.scatter(indices2, poi2, color="red")
#     plt.scatter(indices3, poi3, color="blue")
#     plt.title("设备间熵比和峰值点")
#     plt.show()
#     # plt.pause(1.5)
#     # plt.close()
#     return train_X[:, indices3], train_Y, test_X[:, indices3], test_Y

def Component_FeatureSelect(train_X, train_Y, test_X, test_Y):
    n, a, b = train_X.shape
    train_X = train_X.reshape(n, a * b)  # 将三维数组变形成二维数组
    n2, a, b = test_X.shape
    test_X = test_X.reshape(n2, a * b)
    print(train_X.shape, test_X.shape)
    device_label = np.zeros(n)
    for i in range(int(n / 1000)):
        device_label[i * 1000:(i + 1) * 1000] = i
    print("device_label", device_label)
    begint = time.time()
    total_entropy = calculate_entropy(train_X)  # 每个特征点的信息熵，通过所有样本计算（1*m)
    print("计算完成1,耗时", time.time() - begint, "s")
    class_entropy = calculate_group_entropies(train_X, train_Y)  # 每个类的每个特征点的信息熵（label*m)
    print("计算完成2,耗时", time.time() - begint, "s")
    device_entropy = calculate_group_entropies(train_X, device_label)  # 每个设备的每个特征点的信息熵（device*m)
    print("计算完成3耗时", time.time() - begint, "s")

    class_ratio = np.sum(class_entropy / total_entropy, axis=0)  # 越大越好
    device_ratio = np.sum(device_entropy / total_entropy, axis=0)  # 越小越好
    ratio = class_ratio / device_ratio
    print("熵比", class_ratio.shape, device_ratio.shape, ratio.shape)

    k = 1000
    indices1 = np.argpartition(-class_ratio, k)[:k]  # 选取最大的点
    poi1 = class_ratio[indices1]
    indices2 = np.argpartition(device_ratio, k)[:k]  # 选取最小的点
    poi2 = device_ratio[indices2]
    indices3 = np.argpartition(-ratio, k)[:k]  # 选取最小的点
    poi3 = device_ratio[indices3]
    print(indices3)
    '''
    绘图
    '''
    # t = np.linspace(0, len(train_X[0]), len(train_X[0]))
    # f = np.linspace(0, len(train_X[0][0]), len(train_X[0][0]))
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    plt.title("功耗迹")
    # plt.contourf(t, f, train_X[0].T)
    plt.plot(train_X[0, :])
    plt.subplot(2, 2, 2)
    plt.plot(total_entropy)
    # plt.contourf(t, f, total_entropy.T)
    plt.title("总信息熵")
    plt.subplot(2, 2, 3)
    plt.plot(class_ratio)
    # plt.contourf(t, f, class_ratio.T)
    plt.scatter(indices1, poi1, color="red")
    plt.title("类间熵比和峰值点")
    plt.subplot(2, 2, 4)
    plt.plot(device_ratio)
    # plt.contourf(t, f, device_ratio.T)
    plt.scatter(indices2, poi2, color="red")
    plt.scatter(indices3, poi3, color="blue")
    plt.title("设备间熵比和峰值点")
    plt.show()
    # plt.pause(1.5)
    # plt.close()
    return train_X[:, indices3], train_Y, test_X[:, indices3], test_Y


if __name__ == '__main__':
    pass

    # print(indices3)
    # scipy.io.savemat(DIRECTORY_PREFIX + 'DATA_m(100d200s).mat', {'X': train_X[:, indices3], 'Y': train_Y})
