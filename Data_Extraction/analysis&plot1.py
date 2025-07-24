"""
    @Author : ling chen
    @Create : 2023/03
    @Last modify: 2023/07
    @Description: 对选定功耗轨迹分析并绘图;
                  主要是绘制通过滑动窗口提取多个功耗样本的情形;
"""
import random

import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from src.Configuration_04 import FILENAME, INSTRUCTION_NAME
from src.Data_Extraction.gettime import gettime
import src.Configuration_matplot
# from src.Configuration_data_info import FILENAME, INSTRUCTION_REGNAME

plot_flag = 1  # 控制是否绘图，1绘图，0不绘图
list = [0]  # 指令编号

for k in list:
    print("正在读取指令", INSTRUCTION_NAME[k], " \n文件地址:", FILENAME[k] + ".mat")

    number = [2, 3, 4, 5]  # 样本编号
    for i in number:
        matdata = loadmat(FILENAME[k] + " (" + str(i) + ").mat")  # 读取Mat文件,文件里有两个数组A,B
        A = matdata['A']
        B = matdata['B']
        n = len(A)
        [H_level_t, begin_t] = gettime(k, FILENAME)  # 调用函数gettime,获取指令k的高电平时间段
        H_power = A[H_level_t]
        print("高电平片段长度:", len(H_level_t), "\n")

        if plot_flag == 1:
            t = np.linspace(0, n - 1, n)  # 时间[0, n-1]内均匀取n个点

            # 绘制功率曲线和电平曲线
            plt.figure(figsize=(10, 6))
            ax1 = plt.subplot(2, 1, 1)

            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('功率曲线', color='tab:red')
            ax1.plot(t, A, color='red')
            ax1.tick_params(axis='y', labelcolor='red')

            ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴
            ax2.set_ylabel('电平', color='tab:blue')
            ax2.plot(t, B, color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            # plt.tight_layout()
            # plt.show()

            # 将功率曲线中对应高电平的部分标红
            plt.subplot(2, 1, 2)
            plt.plot(t, A, color='green', linewidth=0.5)
            plt.plot(H_level_t, H_power, color='red', linewidth=0.7)

            # 构造一个滑动窗口
            o = random.randint(-50, 50)  # 随机一个偏移量
            print("本次偏移量:", o)
            new_t = H_level_t + o

            x_begin = H_level_t[0]
            x_end = H_level_t[len(H_level_t) - 1]
            x_begin = x_begin + o
            x_end = x_end + o
            # 标出滑动窗口
            plt.vlines(x_begin, 0.002, 0.018)  # 表示竖线，参数1：x坐标，参数2：y起始坐标，参数3：y终止坐标
            plt.vlines(x_end, 0.002, 0.018)  # 表示竖线，参数1：x坐标，参数2：y起始坐标，参数3：y终止坐标
            plt.hlines(0.018, x_begin, x_end)  # 表示横线，参数1：y坐标，参数2：x起始坐标，参数3：x终止坐标
            plt.hlines(0.002, x_begin, x_end)  # 表示横线，参数1：y坐标，参数2：x起始坐标，参数3：x终止坐标

            plt.show()  # 显示图
