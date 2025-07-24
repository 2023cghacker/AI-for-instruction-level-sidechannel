"""
    @Author : ling chen
    @Create : 2023/02
    @Last modify: 2023/11
    @Description: 将数据集 DATA 进行划分成训练集和测试集，并存入 train_test.mat 文件
"""

import h5py
import numpy as np
import scipy
from scipy.io import loadmat
# from src.Configuration_04 import INSTRUCTION_REGNAME, INSTRUCTION_REGLABEL
# from src.Configuration_data_info import DIRECTORY_PREFIX, INSTRUCTION_NAME, Absolute_address
from sklearn.model_selection import train_test_split

from src.Configuration_04 import DIRECTORY_PREFIX, INSTRUCTION_NAME

# 层次划分
from src.Model_MainCode.Loadmatfile import extract_name
from src.Model_MainCode.accuracy import Accuracy


def divide_by_holdout(labels, fname, flag):
    """
    :param fname: 索引文件名
    :param flag: 控制位，控制是否保存索引文件
    :return: 返回训练集和测试集的索引
    """
    unique_labels = list(set(labels))  # 获取唯一的标签值

    # 构建字典，键为标签值，值为对应的索引列表
    label_indices = {label: [] for label in unique_labels}
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)

    index1 = []
    index2 = []

    # 划分训练集和测试集
    for label, indices in label_indices.items():
        # 将每个标签值对应的索引列表划分为训练集和测试集
        train_idx, test_idx = train_test_split(indices, test_size=0.2)
        index1.extend(train_idx)
        index2.extend(test_idx)

    return index1, index2


# def divide_by_holdout(m, fname, flag):
#     """
#     :param m: 分层采样的层数
#     :param fname: 索引文件名
#     :param flag: 控制位，控制是否保存索引文件
#     :return: 返回训练集和测试集的索引
#     """
#     t = n / m  # 按指令类型（即标签种类）分层
#     index1 = []
#     index2 = []
#     for i in range(m):
#         index1 = np.append(index1, np.arange(t * i, t * i + int(0.8 * t)))
#         index2 = np.append(index2, np.arange(t * i + int(0.8 * t), t * (i + 1)))
#
#     index1 = np.trunc(index1).astype(int)  # 转成整数
#     index2 = np.trunc(index2).astype(int)
#
#     if flag == 1:
#         scipy.io.savemat(fname, {'index1': index1, 'index2': index2})  # 保存索引文件
#     return index1, index2


# 随机划分
def divide_by_rand(Y, fname, flag):
    """
    :param fname: 索引文件名
    :param flag: 控制位，控制是否保存索引文件
    :return: 返回训练集和测试集的索引
    """
    n = len(Y)
    len1 = int(0.8 * n)
    len2 = int(0.2 * n)
    index1 = np.random.rand(len1) * n
    index1 = np.trunc(index1).astype(int)  # 取整
    index2 = np.random.rand(len2) * n
    index2 = np.trunc(index2).astype(int)  # 取整

    if flag == 1:
        scipy.io.savemat(fname, {'index1': index1, 'index2': index2})  # 保存索引文件

    return index1, index2


# 读取索引
def read_index(fname):  # 读取已经生成好的训练集和测试集的种子(索引)
    matdata = loadmat(fname)  # 读取Mat文件, 文件里有两个数组A,B
    index1 = matdata['index1'][0]
    index2 = matdata['index2'][0]
    return index1, index2


def DivideData(X, Y_i, Y_d=-1, TraintestName=False, saveflag=False, DivindexName=None):
    """
    :param X: 数据集输入
    :param Y: 数据集标签
    :param TraintestName: 训练集测试集的存储文件名
    :param DivindexName: 索引文件名
    :return:
    """
    print("before divided:数据集输入规模", X.shape, "数据集标签规模", Y_i.shape)

    '''
        创建训练集和测试集的种子（索引）
        通过索引从总DATA中可以得到训练集和测试集
    '''
    # 产生训练集和测试集的索引，flag=1则保存索引文件，否则不保存
    index1, index2 = divide_by_holdout(Y_i, DivindexName, flag=0)
    # index1, index2 = divide_by_rand(Y,DivindexName, flag=0)
    # index1, index2 = read_index(DivindexName)  # 读取已经生成的索引

    '''
    # 划分训练集和测试集
    '''
    train_X = X[index1, ...]
    train_Y_i = Y_i[index1, ...]
    test_X = X[index2, ...]
    test_Y_i = Y_i[index2, ...]
    if not isinstance(Y_d, bool):
        train_Y_d = Y_d[index1, ...]
        test_Y_d = Y_d[index2, ...]
    print(">----------completed------------>")
    print("before divided:训练集输入规模", train_X.shape, "训练集标签规模", train_Y_i.shape)
    print("               测试集输入规模", test_X.shape, "测试集标签规模", test_Y_i.shape, "\n")

    '''
    # 保存
    '''
    if saveflag:
        scipy.io.savemat(TraintestName,
                         {'train_X': train_X, 'train_Y': train_Y_i, 'test_X': test_X, 'test_Y': test_Y_i})
        print("训练集测试集已经保存在：", extract_name(TraintestName), "\n")

    if not isinstance(Y_d, bool):
        return train_X, train_Y_i, train_Y_d, test_X, test_Y_i, test_Y_d
    else:
        return train_X, train_Y_i, test_X, test_Y_i
