import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

matdata = loadmat("../DataFile/DataSets_Small/DATA.mat")  # 读取Mat文件
X = matdata['X']
# print(X)

# 计算各变量之间的相关系数
X = pd.DataFrame(X)
corr = X.corr(method='pearson')
ax = plt.subplots()  # 调整画布大小
ax = sns.heatmap(corr)  # 画热力图

# 设置刻度字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
