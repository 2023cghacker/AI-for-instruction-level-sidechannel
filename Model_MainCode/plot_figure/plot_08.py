import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.Configuration_05 import DIRECTORY_PREFIX
from src.Model_MainCode.Loadmatfile import loaddevice

# np.random.seed(42)
# X = np.random.randn(100, 3)
# Y = np.random.randint(0, 5, 100)

X, Y_i, Y = loaddevice([1, 2, 3,  5, 6, 7], 100, DIRECTORY_PREFIX + 'DATA_(200d100s).mat')

# 假设 X 是一个 (n, m) 的数据数组，Y 是一个长度为 n 的标签数组
X = np.sum(X, axis=1)

data = pd.DataFrame(X, columns=['Sum_Feature'])
# data = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(m)])
data['Label'] = Y

# 选择要绘制的小提琴图的特征，这里以 Feature_0 为例
feature_to_plot = 'Sum_Feature'

plt.figure(figsize=(10, 6))

# 绘制小提琴图
sns.violinplot(data=data, x='Label', y=feature_to_plot, inner='quartile')

plt.title(f'Violin Plot of {feature_to_plot}')
plt.xlabel('Label')
plt.ylabel(feature_to_plot)
plt.show()
