"""
    @Author : ling chen
    @Create : 2023/03
    @Last modify: 2023/07
    @Description: 将总时间段，高电平对应时间片段，低电平对应时间片段提取出来，
                  绘制高电平触发时的功耗片段，进行分析，以便开展接下来的功耗数据提取工作
"""

import matplotlib.pyplot as plt
import random
from scipy.io import loadmat
import numpy as np
from src.Configuration_data_info import FILENAME
# from src.Configuration_01 import FILENAME
import src.Configuration_matplot


def find_threshold(array):
    # maxnum = np.max(array)
    # minnum = np.min(array)
    # median = (maxnum + minnum) / 2
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


k = 1
print("正在读取文件:", FILENAME[k] + "(15).mat")
matdata = loadmat(FILENAME[k] + ".mat")  # 读取Mat文件,文件里有两个数组A,B
A = matdata['A']
B = matdata['B']
n = len(A)
# print(A)
# print(B)

'''
绘制功率曲线和电平曲线
'''
t = np.linspace(0, n - 1, n)  # 时间[0, n-1]内均匀取n个点
plt.figure(num=1)
plt.plot(t, A, label='Power data')  # 以x为横坐标，y为纵坐标作图，直线/平滑曲线连接
plt.plot(t, B, label='Level signal')  # 散点图
plt.title("功耗曲线和电平曲线")
plt.legend()
plt.show()  # 显示图

'''
绘制功率曲线中的不同部分
'''
h_threshold = find_threshold(B)
print(h_threshold)
h_threshold=(h_threshold[0] + 2 * h_threshold[1] + 3 * h_threshold[2] + 4 * h_threshold[3]) / 10  # 加权平均
print(h_threshold)

H_level_t = np.where(B > h_threshold)[0]  # 找出高电平对应的时间段
L_level_t = np.where(B < h_threshold)[0]  # 找出低电平对应的时间段
H_power = A[H_level_t]  # 找出高电平对应的功耗片段
m = len(H_level_t)
print("高电平部分长度", m)

# begin_t = H_level_t[0]
# H_level_t= np.linspace(begin_t, begin_t + m - 1, m)  # 重新取点
# H_level_t= np.trunc(H_level_t).astype(int)

plt.figure(num=2)
plt.plot(t, A, color='green', linewidth=0.5)
plt.plot(H_level_t, H_power, color='red', linewidth=0.6)  # 将功率曲线中对应高电平的部分标红
plt.show()  # 显示图
