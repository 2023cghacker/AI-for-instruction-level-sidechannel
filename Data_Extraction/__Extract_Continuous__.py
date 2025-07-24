"""
    @Author : ling chen
    @Create : 2023/07
    @Last modify: 2023/07
    @Description: 对长程序采集到的运行功耗进行提取，并且进行重采样，保存到数据集中。
"""
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat
import numpy as np
from src.Configuration_data_info import Absolute_address, DIRECTORY_PREFIX
import src.Configuration_matplot


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


address = Absolute_address + "DataFile/DataSets_Continuous/" + "DATA_program6.mat"  # 存储目录地址前缀
Power = []
n = 0
plot_flag = 0  # 绘图控制符,1表示绘图

for k in range(2, 90):
    # 1.导入mat数据文件
    filename = Absolute_address + 'static/trace_continuous_Instruction/program6/20231109-0001 (' + str(k) + ').mat'
    # filename = Absolute_address + 'static/tracesOfSort/20230711-0001 (' + str(k) + ').mat'
    print(filename)
    matdata = loadmat(filename)  # 读取Mat文件, 文件里有两个数组A,B
    A = matdata['A']
    B = matdata['B']

    # 2.截取中间的高电平片段
    if n == 0:
        h_threshold = find_threshold(B)
        h_threshold = (h_threshold[0] + 2 * h_threshold[1] + 3 * h_threshold[2] + 4 * h_threshold[3]) / 10  # 加权平均
        L_level_t = np.where(B < h_threshold)[0]  # 找出低电平对应的时间段
        low_begin = L_level_t[0]  # 找出第一个低电平开始的时间
        low_end = L_level_t[len(L_level_t) - 1]  # 找出第二个低电平结束的时间
        B1 = B[low_begin:low_end]
        H_level_t = np.where(B1 > h_threshold)[0]  # 找出高电平对应的时间段
        print(H_level_t)
        print("高电平片段长度", len(H_level_t))

    X = A[H_level_t]
    X = np.transpose(X)

    # 3.存储每个样本的功耗迹
    if n == 0:
        Power = X
    else:
        Power = np.append(Power, X, axis=0)
    n = n + 1

    # 4.绘制功耗图和电平图
    if plot_flag == 1:
        t = np.linspace(0, len(A) - 1, len(A))
        plt.subplot(2, 1, 1)
        plt.plot(t, A)
        plt.title("功耗图")
        plt.plot(H_level_t, X[0])
        plt.subplot(2, 1, 2)
        plt.plot(t, B)
        plt.title("电平图")
        plt.show()

# 5.求平均功耗迹，去噪
print("获取的功耗迹规模为:", Power.shape)
Power_mean = sum(Power) / n
plt.plot(H_level_t, Power_mean)
plt.title("平均功耗迹")
plt.show()

# 6.保存
scipy.io.savemat(address, {'Power': Power, 'Power_mean': Power_mean})
