"""
    @Author : ling chen
    @Create : 2023/03
    @Last modify: 2023/07
    @Description: 对采集到的原始数据(含指令信息)处理，分析提取指令（触发高电平对应的功耗片段）,并得到数据集 DATA
"""

import numpy as np
import scipy
from scipy.io import loadmat
from src.Data_Extraction.gettime import gettime
from src.Configuration_04 import FILENAME, INSTRUCTION_NAME, SAMPLES_NUM, INSTRUCTION_LABEL, DIRECTORY_PREFIX

# from src.Configuration_data_info import DIRECTORY_PREFIX, FILENAME, INSTRUCTION_NAME, SAMPLES_NUM, INSTRUCTION_LABEL

'''
    准备工作: 读取和计算一些常量
'''
instruction_name = INSTRUCTION_NAME  # 读取每个指令名
[H_level_t, begin_t] = gettime(0, FILENAME)  # 调用函数gettime，获取第一个指令的单位时间长度，并以此作为基准，以后都使用此长度进行提取指令
m = 750

'''
    开始循环提取样本中的指令（功率片段），
    最后存入X,Y中用于机器学习
'''
Y = []
length = []
n = 0
samples_num = SAMPLES_NUM  # 每条指令采集的样本数

for k in range(len(instruction_name)):  # k表示要对第几个指令（一共21个）进行提取。

    print("\n》》》正在分析提取第", k + 1, '个指令中, 指令名称:', INSTRUCTION_NAME[k], ',样本数:', SAMPLES_NUM, '《《《')
    [H_level_t, begin_t] = gettime(k, FILENAME)  # 调用函数gettime,获取指令k的高电平时间段,起始时间点
    T = round((len(H_level_t) - 2 * m) / m)
    print("该功耗迹高电平时间段为:[", H_level_t[0], ",", H_level_t[len(H_level_t) - 1], "] 长度为:", len(H_level_t),
          "对应指令高电平时间长度为:", len(H_level_t) - 2 * m, ",预计是", T, "周期指令")
    H_level_t = np.linspace(begin_t + m, begin_t + 2 * m - 1, m)  # 重新计算生存所需时间段（开始，结束，采样点数）
    H_level_t = np.trunc(H_level_t).astype(int)
    print("重采样后选择的时间段为:[", H_level_t[0], ",", H_level_t[len(H_level_t) - 1], "] 长度为:", len(H_level_t))

    '''
    开始采集当前指令的样本
    '''
    t = 1
    print("进度条[", end="")
    for i in range(2, 2 + samples_num):  # 对该指令统计其中samples_num条采集的功率样本。
        if i > t * samples_num / 20:
            print("#", end="")  # 不换行
            t = t + 1

        fname = FILENAME[k] + " (" + str(i) + ").mat"
        matdata = loadmat(fname)  # 读取Mat文件, 文件里有两个数组A,B
        A = matdata['A']
        B = matdata['B']

        '''  
        X(i,:)存储了第i次采样中指令触发时对应的功耗;
        X是一个n * m矩阵，n是采样总数，m是指令触发时的时间段长度;
        Y是一个长为n的数组, 存储着该次采样的指令类型（0~20编号）
        '''
        p = A[H_level_t].transpose().tolist()  # 转置并改成list类型
        if n == 0:  # 第一次赋值
            X = p
            Y = INSTRUCTION_LABEL[k]
        else:
            X = np.append(X, p, axis=0)  # 拼接
            Y = np.append(Y, INSTRUCTION_LABEL[k])
        n = n + 1

    print("]")

'''
这个文件最后得到的数据集为： 功率片段X与对应的指令Y 
然后就可以开始跑机器学习了
'''
print(X)
print(Y)
print("数据集规模；", X.shape)
# 将输入X,输出Y,片段长度length都存入DATA.mat
scipy.io.savemat(DIRECTORY_PREFIX + 'DATA_m(750d200s).mat', {'X': X, 'Y': Y})
