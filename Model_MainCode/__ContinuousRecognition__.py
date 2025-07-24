"""
    @Author : ling chen
    @Create : 2023/11
    @Last modify: 2023/11
    @Description: 长程序功耗片段的连续指令识别
"""

import time
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import src.Configuration_matplot
# from src.Configuration_data_info import DIRECTORY_PREFIX, INSTRUCTION_NAME, INSTRUCTION_LABEL,Absolute_address
from src.Configuration_03 import DIRECTORY_PREFIX, INSTRUCTION_NAME, INSTRUCTION_REGNAME, INSTRUCTION_REGLABEL, \
    INSTRUCTION_LABEL, Absolute_address
from src.Model_MainCode.MYAPI_sklearn.BPNN import LoadBPNN


def normalize_array(array):
    # 计算数组的最大值和最小值
    max_value = np.max(array)
    min_value = np.min(array)

    # 将数组归一化
    normalized_array = (array - min_value) / (max_value - min_value)

    return normalized_array


address = Absolute_address + 'DataFile/Datasets_Continuous/DATA_program6.mat'
matdata = loadmat(address)  # 读取Mat文件, 文件里有训练集和测试集
P = matdata['Power']
P = normalize_array(P)
print("程序功耗迹的规模:", P.shape)
total_label = []  # 存储所有样本的预测结果

for ww in range(0, np.size(P, 0)):
    '''
    1.提取数据
    '''
    Power = P[ww]
    print("\n程序功耗片段长度为：", np.shape(Power))

    '''
    2.重采样
    '''
    M = 500  # 一个时钟周期的长度
    m = 500  # 重采样后一个时钟周期的长度
    p = len(Power) // M  # 一共多少时钟周期（取整）
    t = np.linspace(0, p * M - 1, p * m)  # 重新计算生存所需时间段（开始，结束，采样点数）,一个周期设为100个点
    t = np.trunc(t).astype(int)
    Power = Power[t]
    print("\n重采样后，程序功耗片段长度为：", np.shape(Power))
    print("\n程序功耗片段预计含有", p, "个时钟周期，单个时钟周期长度为", m)
    # print(Power)
    # plt.plot(Power)
    # plt.show()

    '''
    3.使用不同大小的滑动窗口将程序片段顺序截取成:
      a)若干个长度为1个周期的样本
      b)若干个长度为2个周期的样本
    '''
    Single_cycle_Power = np.zeros((p, m))  # 单周期指令样本个数为p个
    for i in range(0, p):
        Powerx = Power[i * m:i * m + m]
        Single_cycle_Power[i] = Powerx  # 拼接

    print("\n获得的单周期功耗集规模：", Single_cycle_Power.shape)

    # Double_cycle_Power = np.zeros((p - 1, 2 * m))  # 二周期指令样本个数为p-1个
    # for i in range(0, p - 1):
    #     Powerx = Power[i * m:i * m + 2 * m]
    #     Double_cycle_Power[i] = Powerx  # 拼接
    #
    # print("\n获得的双周期功耗集规模：", Double_cycle_Power.shape)

    '''
    4.对这些单时钟周期样本进行识别
    '''
    print("预测使用的算法是BPNN")
    predict_label = LoadBPNN(DIRECTORY_PREFIX + "1110_12_1Tl_bpnn.m", Single_cycle_Power)
    # predict_label2 = LoadBPNN(DIRECTORY_PREFIX + "1101_233018_2T_bp_model.m", Double_cycle_Power)
    # bpnn,predict_y = CreateBPNN(train_x, train_label, test_x, layersize=1000, t=1000)

    '''
    5.输出当前样本的程序流预测结果
    '''

    print(predict_label)
    print("预测的指令流为：")
    for i in range(len(predict_label)):
        index = predict_label[i]
        print(INSTRUCTION_REGNAME[INSTRUCTION_REGLABEL.index(index)])  # , end="  ===>  ")

    # print("预测的指令流为：")
    # for i in range(len(predict_label2)):
    #     index = predict_label2[i]
    #     print(INSTRUCTION_REGNAME[INSTRUCTION_REGLABEL.index(index)])  # , end="  ===>  ")

    '''
    6.存储预测结果
    '''
    predict_label = predict_label.reshape(1, len(predict_label))
    if ww == 0:
        total_label = predict_label
    else:
        total_label = np.append(total_label, predict_label, axis=0)

"""
7.计算出预测最多的标签（多数投票）
"""
print("==============summary===============")
print(total_label)

# 获取数组的列数
num_cols = total_label.shape[1]

# 初始化一个空列表，用于存储每一列出现次数最多的元素
most_label = []

# 遍历每一列
for col in range(num_cols):
    # 使用NumPy的unique函数来获取每一列的唯一元素以及它们的计数
    unique_elements, counts = np.unique(total_label[:, col], return_counts=True)

    # 找到出现次数最多的元素的索引
    max_count_index = np.argmax(counts)

    # 使用索引获取出现次数最多的元素
    most_frequent_element = unique_elements[max_count_index]

    most_label.append(most_frequent_element)

print(most_label)

for i in range(len(most_label)):
    index = most_label[i]
    print(INSTRUCTION_REGNAME[INSTRUCTION_REGLABEL.index(index)])  # , end="  ===>  ")
