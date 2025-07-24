# 训练集均衡化

import scipy
from scipy.io import loadmat
import numpy as np

from src.Configuration_01 import DIRECTORY_PREFIX

add_pre = DIRECTORY_PREFIX
# ============分层训练测试================
matdata = loadmat(add_pre + 'cosclustering_rS125d_train_test_0')  # 读取Mat文件, 文件里有训练集和测试集
train_X = matdata['train_X']
train_Y = matdata['train_Y']
test_X = matdata['test_X']
test_Y = matdata['test_Y']
n = np.size(train_X, 0)
m = np.size(train_X, 1)  # m=125

train_identifier = np.zeros(n)
train_type = np.zeros(n)
for i in range(n):
    train_identifier[i] = train_Y[i][0]
    train_type[i] = train_Y[i][1]

k = int(np.max(train_type) + 1)
print("训练集一共有", k, "种分类")


'''
按比例对数据集进行均衡，对出现次数多的类别进行删减
'''
# p = np.zeros(k)
# for i in range(k):
#     print("第", i, "类占", np.sum(train_type == i), "个")
#     p[i] = 1 / np.sum(train_type == i)
#
# # 计算出每个类别的提取比例
# p = p / max(p)
# print(p)
#
# t = 0
# m = 0
# new_train_Y = []
# print("进度条[", end="")
# for i in range(n):
#     if i > m * n / 20:
#         print("#", end="")  # 不换行
#         m = m + 1
#     k = train_type[i]
#     if random.random() < p[int(k)]:
#         if t == 0:  # 第一次赋值
#             new_train_Y = [[train_identifier[i], train_type[i]]]
#             new_train_X = [train_X[i]]
#         else:
#             new_train_Y = np.append(new_train_Y, [[train_identifier[i], train_type[i]]], axis=0)  # 拼接
#             new_train_X = np.append(new_train_X, [train_X[i]], axis=0)
#         t = t + 1
# print("]")

'''
手动选择，对出现次数少的类别进行重复采样
'''
type=[1]
for i in type:
    # print(i)
    index = [k for k in range(len(train_type)) if train_type[k] == i]
    # print(train_Y[index])
    print(index)
    new_train_Y = np.append(train_Y,train_Y[index], axis=0)  # 拼接
    new_train_X = np.append(train_X,train_X[index], axis=0)  # 拼接

print(np.size(new_train_X, 0))
print(np.size(new_train_X, 1))

filename = add_pre + "cosclustering_oS125d_train_test_balance.mat"
scipy.io.savemat(filename, {'train_X': new_train_X, 'train_Y': new_train_Y, 'test_X': test_X, 'test_Y': test_Y})
