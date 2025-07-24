"""
    @Author : ling chen
    @Create : 2023/03
    @Last modify: 2023/07
    @Description: 对采集到的原始数据(含指令信息)处理，分析提取指令（触发高电平对应的功耗片段）,并得到数据集 DATA
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import loadmat
from src.Configuration_05 import FILENAME, INSTRUCTION_NAME, SAMPLES_NUM, INSTRUCTION_LABEL, DIRECTORY_PREFIX


def format_number(num):
    # 使用字符串的zfill方法来格式化数字
    return str(num).zfill(4)


def gettime(filename):
    # 对第k个指令进行提取功率片段
    matdata = loadmat(filename)  # 读取Mat文件, 文件里有两个数组A,B
    B = matdata['B'][0]
    # print(B, B.shape)
    # plt.plot(B)
    # plt.show()

    h_threshold = 3  # 设置高电平阈值
    L_level_t = np.where(B < h_threshold)[0]  # 找出低电平对应的时间段
    low_begin = L_level_t[0]  # 找出第一个低电平开始的时间
    low_end = L_level_t[len(L_level_t) - 1]  # 找出第二个低电平结束的时间
    print(low_begin, low_end)
    B1 = B[low_begin:low_end]  # B1里面是先低电平再高电平再低电平

    H_level_t = np.where(B1 > h_threshold)[0]  # 找出高电平对应的时间段
    begin_t = H_level_t[0]  # 高电平的起始时间，即指令片段的起始时间

    return H_level_t, begin_t
def gettimefordevice9(filename):
    # 对第k个指令进行提取功率片段
    matdata = loadmat(filename)  # 读取Mat文件, 文件里有两个数组A,B
    B = matdata['B'][0]
    # print(B, B.shape)
    # plt.plot(B)
    # plt.show()

    h_threshold = 3  # 设置高电平阈值
    H_level_t = np.where(B > h_threshold)[0]  # 找出高电平对应的时间段
    begin_t = H_level_t[0]  # 高电平的起始时间，即指令片段的起始时间

    return H_level_t, begin_t

'''
    开始循环提取样本中的指令（功率片段），
    X存储功耗迹，Y_d存储设备标签，Y_i存储指令标签
'''
Y_d = []
Y_i = []
length = []
n = 0
m = 400

for k in range(len(FILENAME)):
    print("\n》》》正在分析提取第", k + 1, '个文件夹,文件名为:', FILENAME[k], ',采集样本数:', SAMPLES_NUM, '《《《')
    [H_level_t, begin_t] = gettime(FILENAME[k] + "0001.mat")  # 调用函数gettime，获取第一个指令的单位时间长度
    # [H_level_t, begin_t] = gettimefordevice9(FILENAME[k] + "0001.mat")  # 调用函数gettime，获取第一个指令的单位时间长度

    print("该功耗迹高电平时间段为:[", H_level_t[0], ",", H_level_t[len(H_level_t) - 1], "] 长度为:", len(H_level_t))

    '''
    开始采集当前指令的样本
    '''
    t = 1
    print("进度条[", end="")
    for i in range(1, 1 + SAMPLES_NUM):  # 对该指令统计其中samples_num条采集的功率样本。
        if i > t * SAMPLES_NUM / 20:
            print("#", end="")  # 不换行
            t = t + 1

        fname = FILENAME[k] + str(format_number(i)) + ".mat"
        matdata = loadmat(fname)  # 读取Mat文件, 文件里有两个数组A,B
        A = matdata['A'][0]
        B = matdata['B'][0]

        '''第1个指令'''
        idx = list(range(begin_t, begin_t + m))
        p = A[idx].transpose().tolist()  # 转置并改成list类型
        if n == 0:  # 第一次赋值
            X = [p]
            Y_i = INSTRUCTION_LABEL[k * 4]
            Y_d = k // 4 + 1
        else:
            X.append(p)  # 拼接
            Y_i = np.append(Y_i, INSTRUCTION_LABEL[k * 4])  # 指令标签
            Y_d = np.append(Y_d, k // 4 + 1)  # 设备标签
        n = n + 1

        '''第2个指令'''
        idx = list(range(begin_t + m, begin_t + 2 * m))
        p = A[idx].transpose().tolist()  # 转置并改成list类型
        X.append(p)
        Y_i = np.append(Y_i, INSTRUCTION_LABEL[k * 4 + 1])  # 指令标签
        Y_d = np.append(Y_d, k // 4 + 1)  # 设备标签
        n = n + 1

        '''第3个指令'''
        idx = list(range(begin_t + 2 * m, begin_t + 3 * m))
        p = A[idx].transpose().tolist()  # 转置并改成list类型
        X.append(p)
        Y_i = np.append(Y_i, INSTRUCTION_LABEL[k * 4 + 2])  # 指令标签
        Y_d = np.append(Y_d, k // 4 + 1)  # 设备标签
        n = n + 1

        '''第4个指令'''
        idx = list(range(begin_t + 3 * m, begin_t + 4 * m))
        p = A[idx].transpose().tolist()  # 转置并改成list类型
        X.append(p)
        Y_i = np.append(Y_i, INSTRUCTION_LABEL[k * 4 + 3])  # 指令标签
        Y_d = np.append(Y_d, k // 4 + 1)  # 设备标签
        n = n + 1

    print("]")

'''
这个文件最后得到的数据集为： 功率片段X与对应的指令Y
然后就可以开始跑机器学习了
'''
X = np.array(X)
print(Y_i)
print(Y_d)
print("数据集规模；", X.shape)
# 将输入X,输出Y,片段长度length都存入DATA.mat
# scipy.io.savemat(DIRECTORY_PREFIX + 'DATA_(400d1000s).mat', {'X': X, 'Y_i': Y_i,'Y_d':Y_d})
