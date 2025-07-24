"""
    @Author : ling chen
    @Create : 2023/07
    @Last modify: 2023/07
    @Description: 自己拼接生成长程序功耗
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from src.Configuration_01 import INSTRUCTION_NAME, FILENAME, DIRECTORY_PREFIX
from src.Data_Extraction.gettime import gettime


def Situation_01(num):
    """
    拼接多个原始功率片段成一段汇编程序的原始功率片段
    以供长片段的识别
    有噪声片段
    """
    index = np.random.rand(num) * len(instruction_name)
    index = np.trunc(index).astype(int)  # 取整
    print(index)

    Power_segment = []  # 功率片段
    Instruction_id = []  # 指令ID
    Instruction_index = []  # 指令出现的索引
    for k in (index):  # k表示要对第几个指令（一共21个）进行提取。

        [H_level_t, begin_t] = gettime(k)  # 调用函数gettime,获取指令k的高电平时间段,起始时间点
        print("指令名称:", instruction_name[k], "出现位置:", len(Power_segment) + begin_t)
        fname = file_pre[k] + " (" + str(2) + ").mat"
        matdata = loadmat(fname)  # 读取Mat文件, 文件里有两个数组A,B
        A = matdata['A']
        B = matdata['B']

        # 拼接
        Instruction_index = np.append(Instruction_index, len(Power_segment) + begin_t)
        Power_segment = np.append(Power_segment, A)
        Instruction_id = np.append(Instruction_id, k)

    return Power_segment, Instruction_id, Instruction_index


def Situation_02(num):
    """
    拼接多个指令功率片段成一段汇编代码功率片段
    以供长片段的识别
    无噪声片段
    不实际，不如Situation_01有用
    """
    add_pre = DIRECTORY_PREFIX
    matdata = loadmat(add_pre + '125d_DATA')  # 读取Mat文件, 文件里有训练集和测试集
    X = matdata['X']
    Y = matdata['Y'][0]
    n = np.size(X, 0)

    index = np.random.rand(num) * n
    index = np.trunc(index).astype(int)  # 取整
    selected_X = X[index]
    selected_Y = Y[index]

    Power_segment = []  # 功率片段
    Instruction_id = []  # 指令ID
    Instruction_index = []  # 指令出现的索引
    for i in range(num):
        print("指令名称:", instruction_name[selected_Y[i]], "出现位置:", len(Power_segment))
        # 拼接
        Instruction_index = np.append(Instruction_index, len(Power_segment))
        Power_segment = np.append(Power_segment, selected_X[i])
        Instruction_id = np.append(Instruction_id, selected_Y[i])

    return Power_segment, Instruction_id, Instruction_index


file_pre = FILENAME  # 读取文件的地址前缀
instruction_name = INSTRUCTION_NAME  # 读取每个指令名
num = 20  # 设置程序片段中指令的数量
Power_segment, Instruction_id, Instruction_index = Situation_01(num=num)
Instruction_segment = []  # np.trunc(Instruction_segment).astype(int)  # 转成整数
for i in range(num):
    id = int(Instruction_id[i])
    Instruction_segment = Instruction_segment + [instruction_name[id]]

n = len(Power_segment)
t = np.linspace(0, n - 1, n)  # 时间[0, n-1]内均匀取n个点
plt.plot(t, Power_segment)
plt.show()

print("功率片段长度为:", len(Power_segment))
# print("含有指令和出现的索引值为:", Instruction_segment, "\n", Instruction_index)
# print(Instruction_id)

'''
保存
'''
# address = DIRECTORY_PREFIX + 'long_segment_01.mat'
# scipy.io.savemat(address, {'Power_segment': Power_segment, 'Instruction_segment': Instruction_segment,
#                            'Instruction_id': Instruction_id, 'Instruction_index': Instruction_index})
