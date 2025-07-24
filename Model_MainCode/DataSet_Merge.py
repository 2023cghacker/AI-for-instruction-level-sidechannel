import numpy as np
import scipy
from scipy.io import loadmat
from src.Configuration_01 import DIRECTORY_PREFIX

"""
1、导入测试集和训练集
"""
address = DIRECTORY_PREFIX + '125d_DATA'
matdata = loadmat(address)  # 读取Mat文件, 文件里有训练集和测试集
X1 = matdata['X']
n1 = np.size(X1, 0)
Y1 = np.ones(n1)
m = np.size(X1, 1)  # m=125
print("数据集规模；", n1, "*", m)

address = DIRECTORY_PREFIX + 'DATA3'
matdata = loadmat(address)  # 读取Mat文件, 文件里有训练集和测试集
X = matdata['X']
Y = matdata['Y'][0]
n2 = np.size(X, 0)
print("数据集规模；", n2, "*", m)

X = np.append(X, X1, axis=0)
Y = np.append(Y, Y1)
print("合并后的数据集规模：", X.shape, Y.shape)


address = DIRECTORY_PREFIX+'DATA_new.mat'
scipy.io.savemat(address, {'X': X, 'Y': Y})