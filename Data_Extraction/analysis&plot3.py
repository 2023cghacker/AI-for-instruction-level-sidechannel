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
import matplotlib.patches as patches

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
            # 绘制功率曲线和电平曲线
            fig, ax = plt.subplots()
            # plt.plot(t, A, color='green')
            plt.plot(H_level_t[100:1800], H_power[100:1800], color='#154778')

            # 构造一个滑动窗口
            x_begin = 1200  # H_level_t[0]
            x_end = 1800  # H_level_t[len(H_level_t) - 1]

            x_label_position = 1400
            label_text = '1 Cycle'
            # 使用 annotate 方法在指定位置添加标签
            plt.annotate(label_text, (x_label_position,0.016), fontsize=14, family='Times New Roman', )

            # 绘制矩形
            rect = patches.Rectangle(
                (x_begin, -0.012),  # 矩形左下角的坐标
                x_end - x_begin,  # 矩形的宽度
                0.015+0.012,  # 矩形的高度
                linewidth=0,  # 边框宽度
                edgecolor='none',  # 边框颜色
                facecolor='lightblue',  # 矩形填充颜色
                alpha=0.5  # 透明度
            )

            # 将矩形添加到坐标轴中
            ax.add_patch(rect)

            # 标出滑动窗口
            plt.vlines(x_begin, -0.012, 0.015)  # 表示竖线，参数1：x坐标，参数2：y起始坐标，参数3：y终止坐标
            plt.vlines(x_end, -0.012, 0.015)  # 表示竖线，参数1：x坐标，参数2：y起始坐标，参数3：y终止坐标
            plt.hlines(0.015, x_begin, x_end)  # 表示横线，参数1：y坐标，参数2：x起始坐标，参数3：x终止坐标
            plt.hlines(-0.012, x_begin, x_end)  # 表示横线，参数1：y坐标，参数2：x起始坐标，参数3：x终止坐标

            plt.vlines(x_begin-600, -0.012, 0.015,linestyles='--')  # 表示竖线，参数1：x坐标，参数2：y起始坐标，参数3：y终止坐标
            plt.vlines(x_end-600, -0.012, 0.015,linestyles='--')  # 表示竖线，参数1：x坐标，参数2：y起始坐标，参数3：y终止坐标
            plt.hlines(0.015, x_begin-600, x_end-600,linestyles='--')  # 表示横线，参数1：y坐标，参数2：x起始坐标，参数3：x终止坐标
            plt.hlines(-0.012, x_begin-600, x_end-600,linestyles='--')  # 表示横线，参数1：y坐标，参数2：x起始坐标，参数3：x终止坐标

            plt.vlines(x_begin+600, -0.012, 0.015,linestyles='--')  # 表示竖线，参数1：x坐标，参数2：y起始坐标，参数3：y终止坐标
            plt.vlines(x_end+600, -0.012, 0.015,linestyles='--')  # 表示竖线，参数1：x坐标，参数2：y起始坐标，参数3：y终止坐标
            plt.hlines(0.015, x_begin+600, x_end+600,linestyles='--')  # 表示横线，参数1：y坐标，参数2：x起始坐标，参数3：x终止坐标
            plt.hlines(-0.012, x_begin+600, x_end+600,linestyles='--')  # 表示横线，参数1：y坐标，参数2：x起始坐标，参数3：x终止坐标

            plt.vlines(x_begin+1200, -0.012, 0.015,linestyles='--')  # 表示竖线，参数1：x坐标，参数2：y起始坐标，参数3：y终止坐标
            plt.vlines(x_end+1200, -0.012, 0.015,linestyles='--')  # 表示竖线，参数1：x坐标，参数2：y起始坐标，参数3：y终止坐标
            plt.hlines(0.015, x_begin+1200, x_end+1200,linestyles='--')  # 表示横线，参数1：y坐标，参数2：x起始坐标，参数3：x终止坐标
            plt.hlines(-0.012, x_begin+1200, x_end+1200,linestyles='--')  # 表示横线，参数1：y坐标，参数2：x起始坐标，参数3：x终止坐标

            plt.ylim([-0.012, 0.02])
            plt.xlim([H_level_t[100], H_level_t[1850]])
            plt.tight_layout()
            
            ax.spines['top'].set_visible(False)  # 隐藏上边框
            ax.spines['right'].set_visible(False)  # 隐藏右边框
            ax.set_xticks([])  # 隐藏 x 轴的坐标刻度
            ax.set_yticks([])  # 隐藏 y 轴的坐标刻度
            plt.show()  # 显示图
