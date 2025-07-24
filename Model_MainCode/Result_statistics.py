# -*- coding:UTF-8 -*-#
"""
    @filename:Result_statistics.py
    @Author:chenling
    @Create:2024/03/05
"""
"""
    @Author : ling chen
    @Create : 2023/02
    @Last modify: 2023/12
    @Description: 单层标签的训练预测：主程序，这里可以调用 library 中的机器学习 & 深度学习算法
"""
import time
from scipy.io import loadmat
import numpy as np
from src.Configuration_04 import DIRECTORY_PREFIX, INSTRUCTION_NAME, INSTRUCTION_LABEL
from src.Model_MainCode.FeatureEngineering.Component_DomainTransform import Component_DomainTransform
from src.Model_MainCode.FeatureEngineering.Component_FeatureSelect import Component_FeatureSelect
from src.Model_MainCode.FeatureEngineering.Component_Reduction import Component_Reduction
from src.Model_MainCode.FeatureEngineering.Component_Standardized import Component_Standardlized
from src.Model_MainCode.FeatureEngineering.Interpolation import Interpolation_Up_sampling
from src.Model_MainCode.Loadmatfile import loadData, loadTraintest, saveTraintest
from src.Model_MainCode.MYAPI_pytorch.__AttentionCNN_API__ import __AttentionNN_API__
from src.Model_MainCode.MYAPI_pytorch.__BPNN_API__ import __BPNN_API__, __load_bpnnmodel__
from src.Model_MainCode.MYAPI_pytorch.__CNN_API__ import __CNN_API__
from src.Model_MainCode.MYAPI_pytorch.__MLP_API__ import __MLP_API__
from src.Model_MainCode.MYAPI_sklearn.BPNN import CreateBPNN, LoadBPNN
from src.Model_MainCode.__DivideData__ import DivideData
from src.Model_MainCode.accuracy import Accuracy
from src.Model_MainCode.plot_ConfusionMatrix import plot_ConfusionMatrix
import matplotlib.pyplot as plt

# 原始数据集
X, Y = loadData(DIRECTORY_PREFIX + 'DATA_m(750d200s).mat')

TIME = 4  # 总运行次数
acc_list = []
runtime_list = []

'''
# 批量运行，存储结果
'''
for t in range(TIME):
    train_X = X[0:1000 * (t + 1), :]
    train_Y = Y[0:1000 * (t + 1)]
    # train_X = np.append(X[0:3000, :], X[4000:5000, :], axis=0)
    # train_Y = np.append(Y[0:3000], Y[4000:5000], axis=0)
    # train_X = X[0:1000, :]
    # train_Y = Y[0:1000]
    test_X = X[4000:5000, :]
    test_Y = Y[4000:5000]

    begint = time.time()
    train_X, test_X = Component_DomainTransform(train_X, test_X, "CWT")  # 时频域转化组件
    # train_X, train_Y, test_X, test_Y = Component_FeatureSelect(train_X, train_Y, test_X, test_Y)  # 特征选择组件
    # train_X, train_Y, test_X, test_Y = Component_Reduction(train_X, train_Y, test_X, test_Y, "PCA", newD=300)  # 降维组件
    # train_X, train_Y = Component_Standardlized(train_X, train_Y)  # 标准化组件
    # test_X, test_Y = Component_Standardlized(test_X, test_Y)  # 标准化组件

    dict = {"outputdim": 5, "lr": 1e-4, "batch_size": 128, "epoch": 30, "saveflag": False}
    # percentage, predict_Y = __BPNN_API__(train_X, train_Y, test_X, parameters=dict)
    percentage, predict_Y = __CNN_API__(train_X, train_Y, test_X, parameters=dict)
    # percentage, predict_Y = __AttentionNN_API__(train_X, train_Y, test_X, parameters=dict)

    acc = Accuracy(predict_Y, test_Y, 2, INSTRUCTION_NAME, INSTRUCTION_LABEL)  # 准确率
    acc_list.append(acc)
    endt = time.time()
    runtime_list.append(endt - begint)
    print(t, "=> runing time:", endt - begint, "accuracy:", acc)

    del train_X, test_X, percentage

'''
# 绘图
'''

x = [1, 2, 3, 4]
# acc_list = [31.9, 54.5, 57.9, 69.7]  # 原始数据BPNN
# acc_list = [35.9, 51.4, 54.9, 67] #特征提取300
# acc_list = [29.6, 48.8, 51.3, 70.3]  # CWT 750*50 CNN
# acc_list = [30.6, 44.4, 52.3, 70.0]  # CWT 750*50 Attention

# acc_list1 = [42.9, 54.5, 57.9, 69.7]  # 原始数据BPNN
# acc_list2 = [42.9, 61.7, 81.3, 82.3]  # CWT 750*50 CNN
# acc_list3 = [44.3 74.8 82.1 81.2] # CWT 750*50 Attention


plt.figure()
plt.plot(x, acc_list, 'r*')
plt.plot(x, acc_list, lw=1)
# plt.xlabel('number of training device ')
# plt.ylabel('accuracy (%)')
plt.xlabel('训练设备数')
plt.ylabel('准确率(%)')
plt.ylim([20, 100])
plt.grid()
plt.show()
print("finshed")
