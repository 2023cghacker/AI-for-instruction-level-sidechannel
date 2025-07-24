"""
    @Author : ling chen
    @Create : 2023/02
    @Last modify: 2023/12
    @Description: 单层标签的训练预测：主程序，这里可以调用 library 中的机器学习 & 深度学习算法
"""
import time

import scipy
from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
from src.Configuration_05 import DIRECTORY_PREFIX, INSTRUCTION_NAME, INSTRUCTION_LABEL
from src.Model_MainCode.FeatureEngineering.Component_DomainTransform import Component_DomainTransform
from src.Model_MainCode.FeatureEngineering.Component_FeatureSelect import Component_FeatureSelect
from src.Model_MainCode.FeatureEngineering.Component_Reduction import Component_Reduction
from src.Model_MainCode.FeatureEngineering.Component_Standardized import Component_Standardlized
from src.Model_MainCode.FeatureEngineering.Contribution_based_Algorithm import My_featureAlgorithm
from src.Model_MainCode.FeatureEngineering.Interpolation import Interpolation_Up_sampling
from src.Model_MainCode.Loadmatfile import loadData, loadTraintest, saveTraintest, loadData_stm32, loadCWT_stm32, \
    loaddevice
from src.Model_MainCode.MYAPI_pytorch.__AttentionCNN_API__ import __AttentionCNN_API_
from src.Model_MainCode.MYAPI_pytorch.__AttentionDNN_API__ import __AttentionDNN_API__
from src.Model_MainCode.MYAPI_pytorch.__BPNN_API__ import __BPNN_API__, __load_bpnnmodel__
from src.Model_MainCode.MYAPI_pytorch.__CNN_API__ import __CNN_API__
from src.Model_MainCode.MYAPI_pytorch.__LSTM__ import __LSTM_API__
from src.Model_MainCode.MYAPI_pytorch.__MLP_API__ import __MLP_API__
from src.Model_MainCode.MYAPI_pytorch.__ResNet_API__ import __ResNet_API__
from src.Model_MainCode.__DivideData__ import DivideData
from src.Model_MainCode.accuracy import Accuracy
from src.Model_MainCode.plot_ConfusionMatrix import plot_ConfusionMatrix, plot_InstructionMatrix

begint = time.time()

"""
1.导入数据集并划分训练集和测试集：
数据集来自src/DataFile，提取数据集的代码来自src/Data_Extraction，原始示波器数据等来自src/static
DivideData函数来自src/Model_MainCode/_DivideData_
loadTrainstest函数来自src/Model_MainCode/Loadmatfile
# """
# X是功耗迹，Y_i是指令标签，Y_d是设备标签
X, Y_i, Y_d = loaddevice([5], 1000, DIRECTORY_PREFIX + 'DATA_(200d1000s).mat')
# train_X, train_Y_i, train_Y_d = loaddevice([1, 2, 3, 4, 5, 6], 500, DIRECTORY_PREFIX + 'DATA_(200d500s).mat')
# test_X, test_Y_i, test_Y_d = loaddevice([7], 500, DIRECTORY_PREFIX + 'DATA_(200d500s).mat')

# train_X, train_Y_i, train_Y_d = loadCWT_stm32([1,2], DIRECTORY_PREFIX + 'DATA_(50cwt500s)_{')
# test_X, test_Y_i, test_Y_d = loadCWT_stm32([3], DIRECTORY_PREFIX + 'DATA_(50cwt500s)_{')

# X, Y_i, Y_d = loadCWT_stm32([1, 2, 3], DIRECTORY_PREFIX + 'DATA_(cwt500s)_{')
train_X, train_Y_i, train_Y_d, test_X, test_Y_i, test_Y_d = DivideData(X=X, Y_i=Y_i, Y_d=Y_d)
# train_X, train_Y, test_X, test_Y = DivideData(train_X, train_Y, TraintestName, saveflag=False)

"""
3.特征工程：
算法来自src/Model_MainCode/FeatureEngineering
"""
train_X = Component_DomainTransform(train_X, 50)
test_X = Component_DomainTransform(test_X, 50)
# train_X, test_X = My_featureAlgorithm(train_X, train_Y_i, train_Y_d, test_X, K=100)
# train_X = train_X.reshape(len(train_X), 10, 10)
# test_X = test_X.reshape(len(test_X), 10, 10)
# print(train_X.shape, test_X.shape)
# train_X, train_Y_i,test_X, test_Y_i = Component_Reduction(train_X, train_Y_i, test_X, test_Y_i, "LDA", newD=100)  # 降维组件

# 保存
# TraintestName = DIRECTORY_PREFIX + "traintest_m(20x20).mat"
# saveTraintest(TraintestName, train_X, train_Y_i, train_Y_d, test_X, test_Y_i, test_Y_d)

"""
4.算法部分：
算法来自src/Model_MainCode/
MYAPI_Sklearn:KNN，决策树，BPNN，LSTM，KNN，随机森林，SVM
MYAPI_Pytarch:CNN、BPNN、MLP
"""
# predict_Y = CreateBPNN(train_X, train_Y, test_X, layersize=500, t=500, saveflag=True)
# predict_Y = LoadBPNN("../DataFile/DataSets_crossdevice/1219_1226_bpnn.m", test_X)  # 从已有的bp神经网络中导入

# BPNN=epoch200  CNN=epoch30
dict = {"outputdim": 16, "lr": 5e-5, "batch_size": 64, "epoch": 50, "saveflag": False}

# percentage, predict_Y_i = __BPNN_API__(train_X, train_Y_i, test_X, parameters=dict)
# percentage, predict_Y_i = __LSTM_API__(train_X, train_Y_i, test_X, parameters=dict)
# percentage, predict_Y_i = __CNN_API__(train_X, train_Y_i, test_X, parameters=dict)
percentage, predict_Y_i = __ResNet_API__(train_X, train_Y_i, test_X, test_Y_i, parameters=dict)
# percentage, predict_Y_i = __AttentionCNN_API_(train_X, train_Y_i, test_X, parameters=dict)
# percentage, predict_Y_i = __AttentionDNN_API__(train_X, train_Y_i, test_X, parameters=dict)
# percentage, predict_Y = __load_bpnnmodel__(test_X, outputdim=21,
#                                            modeladdress='../DataFile/DataSets_Small/1226_1653_bpnn.pth')

"""
5.结果分析：
使用的函数来自src/Model_MainCode/
"""
Accuracy(predict_Y_i, test_Y_i, 2, INSTRUCTION_NAME, INSTRUCTION_LABEL)  # 准确率
# plot_ConfusionMatrix(predict_Y_i, test_Y_i)  # 分布图
# plot_InstructionMatrix(predict_Y_i, test_Y_i)  # 分布图
endt = time.time()
print("\n==--==--==\nThe running time is: ", round(endt - begint, 2), "second")
