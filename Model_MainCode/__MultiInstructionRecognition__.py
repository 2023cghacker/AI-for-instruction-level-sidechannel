"""
    @Author : ling chen
    @Create : 2023/07
    @Last modify: 2023/11
    @Description: 长程序功耗片段的指令级识别，使用层次模型,
    @tips:此文件暂时作废
"""

import time
import torch
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import src.Configuration_matplot
# from src.Configuration_data_info import DIRECTORY_PREFIX, INSTRUCTION_NAME, INSTRUCTION_LABEL,Absolute_address
from src.Configuration_03 import DIRECTORY_PREFIX, INSTRUCTION_NAME, INSTRUCTION_REGNAME, INSTRUCTION_REGLABEL, \
    INSTRUCTION_LABEL, Absolute_address
from src.Model_MainCode.FeatureEngineering.CWT import CWT, plot_TFdomain
from src.Model_MainCode.MYAPI_pytorch.__CNN_API__ import CNN, __load_model__
from src.Model_MainCode.MYAPI_sklearn.BPNN import LoadBPNN
from src.Model_MainCode.accuracy import Accuracy


def normalize_array(array):
    # 计算数组的最大值和最小值
    max_value = np.max(array)
    min_value = np.min(array)

    # 将数组归一化
    normalized_array = (array - min_value) / (max_value - min_value)
    return normalized_array


'''
1.提取数据
'''
address = Absolute_address + 'DataFile/Datasets_Continuous/DATA_program6.mat'
# address = DIRECTORY_PREFIX + '1T&2T_DATA_l(500d100s).mat'
matdata = loadmat(address)  # 读取Mat文件, 文件里有训练集和测试集
Power = matdata['Power']
print("数据集规模", Power.shape)
m = 500  # 数据维度
plot_flag = 0
model = CNN(15)
model.load_state_dict(torch.load('../DataFile/DataSets_registerbased/1Tm_CNNmodel.pth'))

for ww in range(0, 10):  # np.size(Power, 0)):
    P = Power[ww]

    '''
    2.绘图
    '''
    if plot_flag == 1:
        plt.plot(P)
        plt.show()
        print("程序功耗片段长度为：", np.shape(P))

    '''
    3.使用滑动窗口对功耗片段截取成若干个长度为单个时钟周期的样本
    '''
    Single_cycle_Power = np.zeros((len(P) - m+1, m))
    for i in range(len(Single_cycle_Power)):
        Px = P[i:i + m]
        Single_cycle_Power[i] = Px  # 拼接
    print("获得的单周期功耗集规模：", Single_cycle_Power.shape)
    # plt.plot(Single_cycle_Power[0])
    # plt.show()

    '''
    4.第1层识别，即判断功耗片段是否对齐
    '''
    pass

    '''
    5.第2层识别，识别指令标签
    '''
    input_data = np.zeros((len(Single_cycle_Power), 500, 50))
    for i in range(len(Single_cycle_Power)):
        [cwtmatr, frequencies] = CWT(Single_cycle_Power[i])
        cwtmatr = np.transpose(cwtmatr)
        input_data[i] = cwtmatr


    # plot_TFdomain(Single_cycle_Power[0], frequencies, np.transpose(input_data[0]))
    # input_data = normalize_array(input_data)
    # plot_TFdomain(Single_cycle_Power[0], frequencies, np.transpose(input_data[0]))
    percentage, predict_label = __load_model__(input_data, model)

    # predict_label = LoadBPNN(DIRECTORY_PREFIX + "1217_14_1Tm_bpnn.m", Single_cycle_Power)

    # 输出指令流
    print("当前样本预测的指令流为：")
    print(predict_label)
    for i in range(len(predict_label)):
        index = predict_label[i]
        print(INSTRUCTION_NAME[INSTRUCTION_LABEL.index(index)], end="  =>  ")

    '''
    6.存储每个样本的预测结果
    '''
    predict_label = predict_label.reshape(1, len(predict_label))
    if ww == 0:
        total_label = predict_label
    else:
        total_label = np.append(total_label, predict_label, axis=0)

"""
7.计算出预测最多的标签（多数投票）
"""
print("\n==============summary===============")
print(total_label)
# 获取数组的列数
num_cols = total_label.shape[1]
# 初始化一个空列表，用于存储每一列出现次数最多的元素
most_label = []

# 遍历每一列
for col in range(num_cols):
    # 使用NumPy的unique函数来获取每一列的唯一元素以及它们的计数
    unique_elements, counts = np.unique(total_label[:, col], return_counts=True)
    # print(unique_elements, " ", counts)
    # 找到出现次数最多的元素的索引
    max_count_index = np.argmax(counts)
    # 使用索引获取出现次数最多的元素
    most_frequent_element = unique_elements[max_count_index]
    most_label.append(most_frequent_element)
print(most_label)

np.set_printoptions(suppress=True)  # 关闭Numpy的科学计数法
for i in range(len(most_label)):
    print(i, ":", most_label[i], end=" ")
    index = most_label[i]
    print(INSTRUCTION_NAME[INSTRUCTION_LABEL.index(index)])
    print(percentage[i],"%")
