import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import kstest, shapiro, kurtosis, norm, skew
import src.Configuration_matplot
from src.Configuration_04 import DIRECTORY_PREFIX


def compute_average_vectors(X, Y):
    unique_labels = set(Y)  # 获取唯一的标签值
    n_labels = len(unique_labels)  # 确定标签数量

    average_vectors = np.zeros((n_labels, X.shape[1]))  # 创建一个全零矩阵，用于存储平均向量
    count_vectors = np.zeros(n_labels)  # 创建一个全零矩阵，用于计算每个标签出现的次数

    label_to_index = {label: i for i, label in enumerate(unique_labels)}  # 标签到索引的映射关系

    for i in range(X.shape[0]):
        label = Y[i]  # 获取第 i 个样本的标签
        index = label_to_index[label]  # 获取标签对应的索引
        average_vectors[index] += X[i]  # 将对应标签的样本向量加到该标签对应的位置
        count_vectors[index] += 1  # 将对应标签的计数加1

    for i in range(n_labels):
        if count_vectors[i] != 0:
            average_vectors[i] /= count_vectors[i]  # 计算平均向量

    return average_vectors


def calculate_interclass_differences(X):
    n_samples, n_dimensions = X.shape

    diff_sum = np.zeros(n_dimensions)  # 初始化差异和

    count = 0  # 初始化计数器

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            diff_sum += np.abs(X[i] - X[j])  # 计算样本差异并累加至差异和
            count += 1  # 计算次数累加

    average_diff = diff_sum / count  # 计算平均差异向量
    return average_diff


def calculate_crossdevice_differences(X1, X2):
    n_samples, n_dimensions = X1.shape

    diff_sum = np.zeros(n_dimensions)  # 初始化差异和

    for i in range(n_samples):
        diff_sum += np.abs(X1[i] - X2[i])  # 计算样本差异并累加至差异和

    average_diff = diff_sum / n_samples  # 计算平均差异向量

    return average_diff


def calculate_both_differences(X1, X2):
    n1, n_dimensions = X1.shape
    n2, n_dimensions = X2.shape

    diff_sum = np.zeros(n_dimensions)  # 初始化差异和

    count = 0  # 初始化计数器

    for i in range(n1):
        for j in range(i, n2):
            diff_sum += np.abs(X1[i] - X2[j])  # 计算样本差异并累加至差异和
            count += 1  # 计算次数累加

    average_diff = diff_sum / count  # 计算平均差异向量
    print(count)
    return average_diff


def check_normality(vector):
    # 计算向量的均值和标准差
    mean = np.mean(vector)
    std = np.std(vector)

    stat, p = shapiro((vector - mean) / std)
    print('Shapiro-Wilk Test:')
    print('Statistic:', stat)
    print('p-value:', p)
    if p > 0.05:
        print('The data is approximately normally distributed.')
    else:
        print('The data is not normally distributed.')

    return mean, std


def fit_distribution(vector):
    # 拟合正态分布
    mu, sigma = norm.fit(vector)
    print("正态分布参数-均值:", mu)
    print("正态分布参数-标准差:", sigma)

    # 计算偏度和峰度
    vector_skewness = skew(vector)
    vector_kurtosis = kurtosis(vector)

    print("偏度:", vector_skewness)
    print("峰度:", vector_kurtosis)

    # 绘制直方图和核密度估计曲线
    plt.hist(vector, bins=20, density=True, alpha=0.7, edgecolor='grey', )
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    # plt.plot(x, p, 'k', linewidth=2)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram and Fitted Normal Distribution')
    plt.show()

    return mu, sigma


address = DIRECTORY_PREFIX + 'DATA_m(750d200s).mat'
matdata = loadmat(address)  # 读取Mat文件, 文件里有训练集和测试集
test_X = matdata['X']
test_Y = matdata['Y'][0]
X_1 = test_X[0:1800]
Y_1 = test_Y[0:1800]
X_2 = test_X[4000:5000]
Y_2 = test_Y[4000:5000]

mean_X1 = compute_average_vectors(X_1, Y_1)
mean_X2 = compute_average_vectors(X_2, Y_2)

interclass_differences1 = calculate_interclass_differences(mean_X1)
interclass_differences2 = calculate_interclass_differences(mean_X2)
interclass_differences = (interclass_differences1 + interclass_differences2) / 2
crossdevice_differences = calculate_crossdevice_differences(mean_X1, mean_X2)
crossboth_differences = calculate_both_differences(mean_X1, mean_X2)

plt.hist(interclass_differences2, bins=20, edgecolor='grey', alpha=1, label="跨类差异")
plt.hist(crossdevice_differences, bins=20, edgecolor='black', alpha=0.6, label="跨设备差异")
plt.xlabel('差值')  # 设置x轴标签
plt.ylabel('出现频率')  # 设置y轴标签
plt.title('平均功耗迹之间的绝对差值')  # 设置图表标题
# plt.title('Absolute Difference of Averaged Traces')  # 设置图表标题
# plt.xlabel('D-value')  # 设置x轴标签
# plt.ylabel('Frequency of points')  # 设置y轴标签
plt.xlim(0, 0.0055)
plt.ylim(0, 180)
plt.legend()
plt.show()

fit_distribution(crossdevice_differences)
