from scipy.io import loadmat
from scipy import stats
from scipy.stats import pearsonr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.Configuration_01 import DIRECTORY_PREFIX

add_pre = DIRECTORY_PREFIX
matdata = loadmat(add_pre + '125d_DATA')
# 导入scipy模块
X = matdata['X']
data = X[1]
# 检验是否满足正态分布
u = data.mean()  # 计算均值
std = data.std()  # 计算标准差
print(stats.kstest(data, 'norm', (u, std)))

# 皮尔逊相关系数，【-1，1】绝对值越大越相关
pccs_mean = np.zeros((21, 21))
for K in range(21):
    for K2 in range(K, 21):
        print(K, K2)
        t = 0
        n = 0
        for i in range(0 + K * 111, 110 + K * 111):
            for j in range(0 + K2 * 111, 111 + K2 * 111):
                # print(i,j)
                pccs = pearsonr(X[i], X[j])
                t = t + pccs[0]
                n = n + 1
        pccs_mean[K][K2] = t / n

print(pccs_mean)

ax = plt.subplots()  # 调整画布大小
ax = sns.heatmap(pccs_mean)  # 画热力图
plt.show()

# filename = add_pre + "instruction_pearsonr.mat"
# scipy.io.savemat(filename, {'pccs_matrix': pccs_mean})
# t = 0
# n = 0
# for i in range(0 + 111, 111 + 110):
#     for j in range(i, 111 + 111):
#         # print(i,j)
#         pccs = pearsonr(X[i], X[j])
#         t = t + pccs[0]
#         n = n + 1
# t = t / n
# print(t)
# pccs = np.corrcoef(X[1], X[2])
# print(pccs[0][1])
