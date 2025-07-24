import numpy as np
from scipy.stats import pearsonr


def distance(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum(np.square(x - y)))


def simsilarity(x, y):
    if len(x) != len(y):
        print("向量长度不等")
        return
    a = np.sum(x * y)
    b = np.sum(x * x)
    c = np.sum(y * y)
    return (a / (np.sqrt(b) * np.sqrt(c)))


def KNN(train_X, train_Y, test_X, K):
    n1 = np.size(train_X, 0)
    n2 = np.size(test_X, 0)

    # 设置集合学习对象
    base_X = train_X
    base_Y = train_Y
    # lenth = int(0.5 * n1)
    # index = np.random.rand(lenth) * n1  # 取十分之一用于跑KNN
    # index = np.trunc(index).astype(int)  # 取整
    # base_X = train_X[index]
    # base_Y = train_Y[index]

    predict_Y = np.zeros(n2)
    d = np.zeros(len(base_Y))  # 用于保存距离
    t = 0.01
    print("进度条：[", end="")
    for j in range(np.size(test_X, 0)):
        if j / n2 > t:
            print("#", end="")
            t = t + 0.01

        for i in range(np.size(base_X, 0)):
            d[i] = distance(base_X[i], test_X[j])  # 计算训练集每个样本与新(待测)样本的欧式距离
            # d[i] = simsilarity(base_X[i], test_X[j])  # 计算训练集每个样本与新(待测)样本的余弦相似度
            # d[i] = np.abs(pearsonr(base_X[i], test_X[j])[0])  # 计算训练集每个样本与新(待测)样本的皮尔逊系数

        sorted_id = sorted(range(len(d)), key=lambda x: d[x])  # 从小到大排序
        # sorted_id = sorted(range(len(d)), key=lambda x: d[x], reverse=True)  # 从大到小排序
        K_Y = base_Y[sorted_id[0:K]]
        K_Y = K_Y.astype(int)
        predict_Y[j] = np.argmax(np.bincount(K_Y))  # 取出现频率最高的邻居
        # print(predict_Y[j],test_Y[j])

    print("]")
    return predict_Y
    print(predict_Y[j])


x = [1, 1.21, 1, 1, 45.1, 6.2]
y = [2, 2, 2, 2, 564, 12.4]
x = np.array(x)
y = np.array(y)
c = simsilarity(x, y)
print(c)
# 25521.7 2076.9141 318265.76
# 0.9926710874326409
