"""
    @Author : ling chen
    @Create : 2023/03
    @Last modify: 2023/07
    @Description: 对原始数据处理，通过滑动窗口截取功耗数据样本群，可以扩展样本的维度空间
                  得到包含 对齐功耗 and非对齐功耗 的数据集
"""
import random
import numpy as np
import scipy
from scipy.io import loadmat
from src.Data_Extraction.gettime import gettime
from src.Configuration_data_info import FILENAME, INSTRUCTION_REGNAME, DIRECTORY_PREFIX,INSTRUCTION_REGLABEL


def fixed_sliding(X, Y, offset, n, o):
    """
    @根据输入的一个固定偏移量，得到滑动窗口，然后对功耗片段进行截取，加入数据集 X,Y中
    :param X: 属性矩阵（输入矩阵）
    :param Y: 标签向量（输出向量)
    :param offset: 存储每次偏移量的数组
    :param n: 当前行数（样本数）
    :param o: 本次偏移量
    :return: (X, Y, offset, n)
    """
    # 1.本次偏移量
    print("本次偏移量:", o)

    # 2.构造一个滑动窗口,即截取的时间段
    new_t = H_level_t + o

    '''  
      3.获取样本加入数据集
        X是功耗矩阵，即输入矩阵，Y是标签向量，即输出向量
    '''
    p = A[new_t].transpose().tolist()  # 转置并改成list类型
    if n == 0:  # 第一次赋值
        X = p
        Y = k
        offset = o
    else:
        X = np.append(X, p, axis=0)  # 拼接
        Y = np.append(Y, k)
        offset = np.append(offset, o)
    n = n + 1

    return (X, Y, offset, n)


def rand_sliding(X, Y, offset, n):
    """
    @随机生成一个偏移量，得到滑动窗口，然后对功耗片段进行截取，加入数据集 X,Y中
    :param X: 属性矩阵（输入矩阵）
    :param Y: 标签向量（输出向量)
    :param offset: 存储每次偏移量的数组
    :param n: 当前行数（样本数）
    :return: (X, Y, offset, n)
    """
    # 1.随机得到本次偏移量
    o = random.randint(-50, 50)  # 随机一个偏移量
    # print("本次偏移量:", o)

    # 2.构造一个滑动窗口,即截取的时间段
    new_t = H_level_t + o

    '''  
      3.获取样本加入数据集
        X是功耗矩阵，即输入矩阵，Y是标签向量，即输出向量
    '''
    p = A[new_t].transpose().tolist()  # 转置并改成list类型
    if n == 0:  # 第一次赋值
        X = p
        Y = k
        offset = o
    else:
        X = np.append(X, p, axis=0)  # 拼接
        Y = np.append(Y, k)
        offset = np.append(offset, o)
    n = n + 1

    return (X, Y, offset, n)


def order_sliding1(X, Y, offset, n):
    """
     @按序从给定区间生成一系列偏移量得到一系列滑动窗口,然后对功耗片段进行截取，加入数据集 X,Y中
     对齐和未对齐的标签都是原始标签
     :param X: 属性矩阵（输入矩阵）
     :param Y: 标签向量（输出向量)
     :param offset: 存储每次偏移量的数组
     :param n: 当前行数（样本数）
     :return: (X, Y, offset, n)
     """
    for o in range(-10, 10):
        # 1.按序生成偏移量
        # print("本次偏移量:", o)

        # 2.构造一个滑动窗口,即截取的时间段
        new_t = H_level_t + o

        '''  
          3.获取样本加入数据集
            X是功耗矩阵，即输入矩阵，Y是标签向量，即输出向量
        '''
        p = A[new_t].transpose().tolist()  # 转置并改成list类型
        if n == 0:  # 第一次赋值
            X = p
            Y = ins_label[k]
            offset = o
        else:
            X = np.append(X, p, axis=0)  # 拼接
            Y = np.append(Y, ins_label[k])
            offset = np.append(offset, o)
        n = n + 1

    return (X, Y, offset, n)


def order_sliding2(X, Y, offset, n):
    """
     @按序从给定区间生成一系列偏移量得到一系列滑动窗口,然后对功耗片段进行截取，加入数据集 X,Y中
     对齐的标签是指令的序号，未对齐的标签是-1
     :param X: 属性矩阵（输入矩阵）
     :param Y: 标签向量（输出向量)
     :param offset: 存储每次偏移量的数组
     :param n: 当前行数（样本数）
     :return: (X, Y, offset, n)
     """
    for o in range(-10, 10):  # 按序生成偏移量
        # print("本次偏移量:", o)
        new_t = H_level_t + o
        '''  
        X(i,:)存储了第i次采样中指令触发时对应的功耗;
        X是一个n * m矩阵，n是采样总数，m是指令触发时的时间段长度;
        Y是一个长为n的数组, 存储着该次采样的指令类型（0~20编号）
        '''
        p = A[new_t].transpose().tolist()  # 转置并改成list类型
        if n == 0:  # 第一次赋值
            X = p
            # Y = k
            if o == 0:  # 无偏移时
                Y = ins_label[k]
            else:
                Y = -1
            offset = o
        else:
            X = np.append(X, p, axis=0)  # 拼接
            # Y = np.append(Y, k)
            if o == 0:  # 无偏移时
                Y = np.append(Y, ins_label[k])
            else:
                Y = np.append(Y, -1)
            offset = np.append(offset, o)
        n = n + 1

    return (X, Y, offset, n)


def order_sliding3(X, Y, offset, n):
    """
     @按序从给定区间生成一系列偏移量得到一系列滑动窗口,然后对功耗片段进行截取，加入数据集 X,Y中
     按序从[-50，50]生成偏移量得到滑动窗口,对齐的标签是1，未对齐的标签是0
     对齐的标签是指令的序号，未对齐的标签是-1
     :param X: 属性矩阵（输入矩阵）
     :param Y: 标签向量（输出向量)
     :param offset: 存储每次偏移量的数组
     :param n: 当前行数（样本数）
     :return: (X, Y, offset, n)
     """
    for o in range(-250, 250, 5):  # 按序生成偏移量
        # print("本次偏移量:", o)
        new_t = H_level_t + o
        '''  
        X(i,:)存储了第i次采样中指令触发时对应的功耗;
        X是一个n * m矩阵，n是采样总数，m是指令触发时的时间段长度;
        Y是一个长为n的数组, 存储着该次采样的指令类型（0~20编号）
        '''
        p = A[new_t].transpose().tolist()  # 转置并改成list类型
        if n == 0:  # 第一次赋值
            X = p
            if -10 < o and o < 10:  # 无偏移时
                Y = 1
            else:
                Y = 0
            offset = o
        else:
            X = np.append(X, p, axis=0)  # 拼接
            if o == 0:  # 无偏移时
                Y = np.append(Y, 1)
            else:  # 有偏移时
                Y = np.append(Y, 0)
            offset = np.append(offset, o)

        n = n + 1

    return (X, Y, offset, n)


'''
    准备工作: 读取和计算一些常量
'''
file_pre = FILENAME  # 读取文件的地址前缀
M = 500  # 指令单周期长度 
m = 500  # 重采样指令单周期长度
ins_name = INSTRUCTION_REGNAME
ins_label = INSTRUCTION_REGLABEL

'''
    开始循环提取样本中的指令（功率片段），
    最后存入X,Y中用于机器学习
'''
X_1 = []
Y_1 = []
X_2 = []
Y_2 = []
offset = []
n1 = 0
n2 = 0
samples_num = 50  # SAMPLES_NUM  # 每条指令采集的样本数

for k in range(len(ins_name)):  # k表示要对第几个指令类进行提取。

    print("\n》》》正在分析提取第", k + 1, '个指令中,指令名称:', ins_name[k], '《《《')
    [H_level_t, begin_t] = gettime(k, FILENAME)  # 调用函数gettime,获取指令k的高电平时间段,起始时间点
    T = round((len(H_level_t) - 2 * M) / M)
    print("该功耗迹高电平时间段为:[", H_level_t[0], ",", H_level_t[len(H_level_t) - 1], "] 长度为:", len(H_level_t),
          "对应指令高电平时间长度为:", len(H_level_t) - 2 * m, ",预计是", T, "周期指令")
    H_level_t = np.linspace(begin_t+M-1, begin_t + M * (T+1) - 1, m * T)  # 重新计算生存所需时间段（开始，结束，采样点数）
    H_level_t = np.trunc(H_level_t).astype(int)
    print("重采样后的基准时间段为:[", H_level_t[0], ",", H_level_t[len(H_level_t) - 1], "] 长度为:", len(H_level_t))

    '''
    开始采集当前指令的样本
    '''
    print("当前采用了滑动窗口[-10,10]")
    print("进度条[", end="")
    t = 1
    for i in range(2, 2 + samples_num):  # 对该指令统计其中samples_num条采集的功率样本。
        if i > t * samples_num / 20:
            print("#", end="")  # 不换行
            t = t + 1

        fname = file_pre[k] + " (" + str(i) + ").mat"
        matdata = loadmat(fname)  # 读取Mat文件, 文件里有两个数组A,B
        A = matdata['A']
        B = matdata['B']

        if T == 1:  # 单周期指令的存储变量

            X_1, Y_1, offset, n1 = order_sliding1(X_1, Y_1, offset, n1)

        if T == 2:  # 双周期指令的存储变量

            X_2, Y_2, offset, n2 = order_sliding1(X_2, Y_2, offset, n2)

        # 生成固定偏移量的滑动窗口
        # X, Y, offset, n = fixed_sliding(X, Y, offset, n, o=2)
        # 随机生成滑动窗口
        # X, Y, offset, n = rand_sliding(X, Y, offset, n)

    print("]")

'''
这个文件最后得到的数据集为： 功率片段X与对应的指令Y 
然后就可以开始跑机器学习了
'''
print("单周期指令数据集规模；", X_1.shape)
print("双周期指令数据集规模；", X_2.shape)

# 将输入X,输出Y,片段长度length都存入DATA.mat
scipy.io.savemat(DIRECTORY_PREFIX + '1T&2T_DATA_m(sliding100s).mat', {'X_1': X_1, 'Y_1': Y_1, 'X_2': X_2, 'Y_2': Y_2})
