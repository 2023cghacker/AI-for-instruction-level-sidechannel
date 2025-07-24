"""
    @Author : ling chen
    @Create : 2023/03
    @Last modify: 2023/07
    @Description: 对选定数据集分析并绘图;
                  绘制所有功耗样本的平均图形;
"""
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from src.Configuration_01 import FILENAME, INSTRUCTION_NAME, SAMPLES_NUM


plot_flag = 1  # 控制是否绘图，1绘图，0不绘图

n = 2331
A_mean = np.zeros(417)
print(A_mean)
for k in range(0, len(INSTRUCTION_NAME)):
    print("正在读取第", k + 1, "个指令", INSTRUCTION_NAME[k], "\n")
    for i in range(2, 2 + SAMPLES_NUM):  # 对该指令统计其中samples_num条采集的功率样本。
        fname = FILENAME[k] + " (" + str(i) + ").mat"
        matdata = loadmat(fname)  # 读取Mat文件, 文件里有两个数组A,B
        A = matdata['A']
        B = matdata['B']
        A = A.transpose().tolist()  # 转置并改成list类型
        A_mean = A_mean + A[0]

A_mean = A_mean / n
print(A_mean)
plt.plot(A_mean)
plt.show()
