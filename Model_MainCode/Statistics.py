import numpy as np
from scipy.io import loadmat

from src.Configuration_01 import DIRECTORY_PREFIX

add_pre = DIRECTORY_PREFIX
matdata = loadmat(add_pre + '1500d_DATA.mat')  # 读取Mat文件, 文件里有两个数组A,B
X = matdata['X']
Y = matdata['Y'][0]
n = np.size(X, 0)
m = np.size(X, 1)  # m=125
t = np.linspace(1, 125, 125)  # 时间[1, n]内均匀取n个点

#得到每个指令的统计学指标
mean_x = np.zeros(21)
var_x = np.zeros(21)
for i in range(21):
    x = sum(X[i * 111:i * 111 + 111]) / 111
    # print(x,"\n")
    mean_x[i] = np.mean(x) # *np.ones(125)
    var_x[i] = np.var(x) # *np.ones(125)
    # mean_topx=0.012*np.ones(125)
    #
    # plt.figure(num=1)
    # line1, = plt.plot(t, x, color='red', linewidth=0.5)
    # line2, = plt.plot(t, mean_x, color='blue', linewidth=1)
    # line2, = plt.plot(t, mean_topx,  linewidth=1)
    # plt.show()  # 显示图
    print((mean_x[i]-0.01)*100000,var_x[i]*1000000)

# 将多条曲线绘制在一张图上
# for i in range(125):
#     x = X[i]
#     plt.figure(num=1)
#     plt.plot(t, x, color='red', linewidth=0.5)
#
# plt.plot(t, 0.012*np.ones(125),  linewidth=1)
# plt.show()  # 显示图

# max_x = np.zeros(n)
# min_x = np.zeros(n)
# mean_x = np.zeros(n)
# var_x=np.zeros(n)
#
# for i in range(n):
#     x = X[i]
#     x = sorted(x)  # 从小到大排序
#     max_x[i] = np.mean(x[100:125])  # 最大的25个瞬时功率取平均
#     min_x[i] = np.mean(x[0:25])  # 最小的25个瞬时功率取平均
#     mean_x[i] = np.mean(x)  # 所有功率取平均
#     var_x[i]=np.var(x)
#
#     # 扩大数量级
#     max_x[i] = (max_x[i] - 0.01) * 100
#     min_x[i] = (min_x[i] - 0.01) * 100
#     mean_x[i] = (mean_x[i] - 0.01) * 100
#     print(max_x[i], min_x[i], mean_x[i], "\n")
#
#
# matdata = loadmat("../DataFile/1500d_DATA.mat")  # 读取Mat文件
# X = matdata['X']
# m = np.size(X, 1)
# new_X = np.zeros((n, m + 3))
# for i in range(n):
#     for j in range(m):
#         new_X[i][j] = X[i][j]
#         new_X[i][m] = max_x[i]
#         new_X[i][m + 1] = min_x[i]
#         new_X[i][m + 2] = mean_x[i]
#
# scipy.io.savemat("../DataFile/128d_DATA.mat", {'X': new_X, 'Y': Y})

