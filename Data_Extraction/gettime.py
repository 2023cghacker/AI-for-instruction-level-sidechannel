"""
    @Author : ling chen
    @Create : 2023/03
    @Last modify: 2023/07
    @Description: 将 __FindTime__.py 的重要功能进行封装，作为一个函数被 data_preprocessing 调用
"""

from scipy.io import loadmat
import numpy as np


def find_threshold(array):
    # 对数组进行降序排序
    sorted_array = sorted(array, reverse=False)

    # 找到不重复的三个值
    unique_three = []
    for num in sorted_array:
        if num not in unique_three:
            unique_three.append(num)
        if len(unique_three) == 5:
            break

    unique_three.remove(unique_three[0])
    unique_three = np.array(unique_three)
    unique_three = unique_three.flatten()

    return unique_three


def gettime(k, FILENAME):
    # 对第k个指令进行提取功率片段
    filename = FILENAME[k] + ".mat"
    matdata = loadmat(filename)  # 读取Mat文件, 文件里有两个数组A,B
    B = matdata['B']

    h_threshold = find_threshold(B)
    h_threshold = (h_threshold[0] + 2 * h_threshold[1] + 3 * h_threshold[2] + 4 * h_threshold[3]) / 10  # 加权平均

    L_level_t = np.where(B < h_threshold)[0]  # 找出低电平对应的时间段
    low_begin = L_level_t[0]  # 找出第一个低电平开始的时间
    low_end = L_level_t[len(L_level_t) - 1]  # 找出第二个低电平结束的时间
    B1 = B[low_begin:low_end] #B1里面是先低电平再高电平再低电平


    H_level_t = np.where(B1 > h_threshold)[0]  # 找出高电平对应的时间段
    begin_t = H_level_t[0]  # 高电平的起始时间，即指令片段的起始时间

    return (H_level_t, begin_t)
