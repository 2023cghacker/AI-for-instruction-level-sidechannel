"""
    @Author : ling chen
    @Create : 2023/07
    @Last modify: 2023/07
    @Description: 对采集到的原始数据(含指令信息，寄存器信息 )处理，分析提取指令（触发高电平对应的功耗片段）,并得到数据集 DATA
"""

import numpy as np
import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt
from src.Data_Extraction.gettime import gettime
from src.Configuration_data_info import DIRECTORY_PREFIX, FILENAME, SAMPLES_NUM, INSTRUCTION_REGNAME, \
    INSTRUCTION_REGLABEL

"""
    准备工作: 读取和定义一些常量,变量
    Y_T:T周期指令的标签向量 
    n_T:T周期指令的样本总数
"""
file_pre = FILENAME  # 文件的地址前缀
M = 500  # 指令单周期长度
m = 500  # 重采样指令单周期长度
Y_1 = []
Y_2 = []
n_1 = 0
n_2 = 0
"""
    循环提取各个指令的数据 并生成数据集
"""
for k in range(len(INSTRUCTION_REGNAME)):  # k表示要对第k个指令进行提取。
    print("\n》》》正在分析提取第", k + 1, '个指令中, 指令名称:', INSTRUCTION_REGNAME[k], ',样本数:', SAMPLES_NUM, '《《《')

    [H_level_t, begin_t] = gettime(k, FILENAME)  # 调用函数gettime,获取指令k的mat文件中高电平时间段,起始时间点
    T = round((len(H_level_t) - 2 * M) / M)
    print("该功耗迹高电平时间段为:[", H_level_t[0], ",", H_level_t[len(H_level_t) - 1], "]  长度为", len(H_level_t),
          ", 其中目标指令时间长度为", len(H_level_t) - 2 * M, ", 预计为", T, "周期指令")

    H_level_t = np.linspace(begin_t + M - 1, begin_t + M * (T + 1) - 1, m * T)  # 重新计算生存所需时间段（开始，结束，采样点数）
    H_level_t = np.trunc(H_level_t).astype(int)
    print("重采样后选择的时间段为:[", H_level_t[0], ",", H_level_t[len(H_level_t) - 1], "]  长度为", len(H_level_t))

    '''
    开始采集当前指令的样本
    '''
    print("进度条[", end="")
    t = 1
    for i in range(2, 2 + SAMPLES_NUM):  # 对该指令统计其中samples_num条采集的功率样本。
        if i > t * SAMPLES_NUM / 20:
            print("#", end="")  # 不换行
            t = t + 1

        fname = file_pre[k] + " (" + str(i) + ").mat"
        matdata = loadmat(fname)  # 读取Mat文件, 文件里有两个数组A,B
        A = matdata['A']
        B = matdata['B']

        '''
        X(i,:)存储了第i次采样中指令触发时对应的功耗;
        X是一个n * m矩阵，n是采样总数，m是指令触发时的时间段长度;
        Y是一个长为n的数组, 存储着该次采样的指令类型（编号），作为机器学习的标签;
        '''
        p = A[H_level_t].transpose().tolist()  # 将功耗片段提取出来，转置并改成list类型

        if T == 1:  # 单周期指令的存储变量
            if n_1 == 0:  # 第一次赋值
                X_1 = p
                Y_1 = INSTRUCTION_REGLABEL[k]
            else:
                X_1 = np.append(X_1, p, axis=0)  # 拼接
                Y_1 = np.append(Y_1, INSTRUCTION_REGLABEL[k])
            n_1 = n_1 + 1

        if T == 2:  # 双周期指令的存储变量
            if n_2 == 0:  # 第一次赋值
                X_2 = p
                Y_2 = INSTRUCTION_REGLABEL[k]
            else:
                X_2 = np.append(X_2, p, axis=0)  # 拼接
                Y_2 = np.append(Y_2, INSTRUCTION_REGLABEL[k])
            n_2 = n_2 + 1

    # 采完一条指令
    print("]")
print("\n》》》采集结束《《《\n")

'''
这个文件最后需要得到的数据集为： 功率轨迹矩阵X与对应的指令标签向量Y
然后就可以开始跑机器学习了
'''
t = np.linspace(0, m - 1, m)
plt.plot(t, X_1[1])
plt.show()
print("单周期指令数据集规模；", X_1.shape)
print("双周期指令数据集规模；", X_2.shape)

# 将输入X,输出Y,片段长度length都存入DATA.mat
scipy.io.savemat(DIRECTORY_PREFIX + '1T&2T_DATA_m(500d100s).mat', {'X_1': X_1, 'Y_1': Y_1, 'X_2': X_2, 'Y_2': Y_2})
