import random
from scipy.io import loadmat
import numpy as np
from src.Configuration_01 import DIRECTORY_PREFIX


def find_index(a, x):  # 在数组a中寻找x
    for i in range(np.size(a, 0)):
        if a[i][0] == x[0]:
            # print(i)
            return i


def cal_distance(node, centor):  # 计算欧式距离
    node = np.array(node)
    centor = np.array(centor)
    return np.sqrt(np.sum(np.square(node - centor)))


def random_centor(data, k):  # 随机K个中心点
    data = list(data)
    return random.sample(data, k)


def get_cluster(data, centor):
    cluster_dict = dict()
    k = len(centor)
    for node in data:
        cluster_class = -1
        min_distance = float('inf')
        for i in range(k):
            dist = cal_distance(node, centor[i])  # 计算每个点到第i个中心点的距离
            if dist < min_distance:
                cluster_class = i
                min_distance = dist
        if cluster_class not in cluster_dict.keys():  # 初始化集合
            cluster_dict[cluster_class] = []
        cluster_dict[cluster_class].append(node)
    return cluster_dict


def get_centor(cluster_dict, k):  # 计算新中心点：根据簇类坐标平均值
    new_centor = []
    for i in range(k):
        centor = np.mean(cluster_dict[i], axis=0)
        new_centor.append(centor)
    return new_centor


def k_means(data, k):
    centor = random_centor(data, k)  # 初始化k个中心点
    cluster_dict = get_cluster(data, centor)  # 获取聚类后的簇
    new_centor = get_centor(cluster_dict, k)  # 计算新中心点

    while cal_distance(new_centor, centor) > 0:  # 循环聚类和计算新中心点
        centor = new_centor  # 更新老中心点
        cluster_dict = get_cluster(data, centor)
        new_centor = get_centor(cluster_dict, k)  # 计算新中心点
    return cluster_dict, centor


add_pre = DIRECTORY_PREFIX
matdata = loadmat(add_pre + 'instruction_cossim_125d')
cos_sim_mean = matdata['cos_sim_mean'][0]  # 取第一行，也就是以第一个指令为基准
cos_sim_var = matdata['cos_sim_var'][0]  # 取第一行，也就是以第一个指令为基准的余弦相似度
print(np.arccos(cos_sim_mean)/3.14159*180)
# print(cos_sim_mean,cos_sim_var)

# 归一化
# cos_sim_mean = (cos_sim_mean - np.min(cos_sim_mean)) / (np.max(cos_sim_mean) - np.min(cos_sim_mean))
# cos_sim_var = (cos_sim_var - np.min(cos_sim_var)) / (np.max(cos_sim_var) - np.min(cos_sim_var))
# print(cos_sim_mean, cos_sim_var)
data = np.zeros((len(cos_sim_mean), 2))
for i in range(len(cos_sim_mean)):
    data[i][0] = cos_sim_mean[i]
    data[i][1] = cos_sim_var[i]
print(data)
# data = np.arccos(cos_sim_mean) / 3.141592 * 180
# data = data.tolist()
# print(data)
# data = np.array([[1, 1, 1], [2, 2, 2], [1, 2, 1], [9, 8, 7], [7, 8, 9], [8, 9, 7]])

# K-means聚类
K = 4
a, centor = k_means(data, K)
print("\n中心点为：", centor)
for k in range(K):
    print("\n第", k, "类簇：", end=" ")
    for i in range(len(a[k])):
        node = a[k][i]
        # print(node, end=" ")
        index = find_index(data, node)
        # print(index, INSTRUCTION_NAME[index],end=",   ")
        print(index, end=",   ")

# print("\n")
# x=np.array([0.9602071214192226,0])
# # print(x[0])
# find_index(data, x)
