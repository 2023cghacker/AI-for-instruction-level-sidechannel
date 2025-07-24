import numpy as np
import scipy
from scipy.io import loadmat


def standardize(D):
    """
    对二维数组D进行标准化处理，使其均值为0，标准差为1。

    参数:
    D -- 二维numpy数组，形状为(n_samples, n_features)

    返回:
    D_std -- 标准化后的二维numpy数组，形状与D相同
    """
    # 计算每个特征的均值
    mean_vals = np.mean(D, axis=0)
    # 计算每个特征的标准差
    std_vals = np.std(D, axis=0, ddof=1)  # 使用ddof=1以获得样本标准差
    # 避免除以零（虽然理论上不应该发生，但好的做法是添加一个小的常数如epsilon）
    # std_vals[std_vals == 0] = 1.e-7
    # 标准化处理
    D_std = (D - mean_vals) / std_vals
    return D_std


def extract_name(text):
    index = text.find("DataFile/")
    if index != -1:
        result = text[index:]  # +4 是为了排除 "src/" 这部分
        return result
    else:
        return "地址可能出错"


def loadData(address, name1='X', name2='Y'):
    matdata = loadmat(address)  # 读取Mat文件
    X = matdata[name1]
    Y = matdata[name2][0]
    print("mat文件地址:", extract_name(address))
    print(" >数据集输入规模", X.shape, "数据集标签规模", Y.shape, "\n")
    return X, Y


def loadData_stm32(address):
    matdata = loadmat(address)  # 读取Mat文件
    X = matdata['X']
    Y_i = matdata['Y_i'][0]
    Y_d = matdata['Y_d'][0]
    print("mat文件地址:", extract_name(address))
    print(" >数据集输入规模", X.shape, "\n")
    return X, Y_i, Y_d


def loaddevice(device_list, num, address):
    matdata = loadmat(address)  # 读取Mat文件
    X = matdata['X']
    Y_i = matdata['Y_i'][0]
    Y_d = matdata['Y_d'][0]
    selected_X = []
    idx = []
    num = num * 16
    flag = False
    for d in device_list:
        idx += list(range((d - 1) * num, d * num))
        if flag:
            new_X = X[list(range((d - 1) * num, d * num))]
            new_X = X[list(range((d - 1) * num, d * num))]+original_X
            selected_X = np.vstack((selected_X, new_X))
        else:
            original_X = X[list(range(0 * num, 1 * num))]
            selected_X = X[list(range((d - 1) * num, d * num))]+original_X
            flag = True

        print(f"shape={selected_X.shape}")

    Y_i = Y_i[idx]
    Y_d = Y_d[idx]
    # print(f"shape={selected_X.shape}")
    # print(idx,len(idx))

    return selected_X, Y_i, Y_d


def loadCWT_stm32(device_list, address):
    device_num = len(device_list)
    X = np.zeros((device_num * 16 * 500, 200, 50))
    Y_i = np.zeros(device_num * 16 * 500)
    Y_d = np.zeros(device_num * 16 * 500)
    print(X.shape)
    idx = 0
    for device_id in device_list:
        for i in range((device_id - 1) * 8, device_id * 8):
            print(i, idx * 1000, (idx + 1) * 1000)
            matdata = loadmat(address + str(i) + "}.mat")  # 读取Mat文件
            X[idx * 1000:(idx + 1) * 1000] = matdata['X']
            Y_i[idx * 1000:(idx + 1) * 1000] = matdata['Y_i'][0]
            Y_d[idx * 1000:(idx + 1) * 1000] = matdata['Y_d'][0]
            idx = idx + 1

    return X, Y_i, Y_d


def loadTraintest(address, name1='train_X', name2='train_Y', name3='test_X', name4='test_Y'):
    matdata = loadmat(address)  # 读取Mat文件
    train_X = matdata[name1]
    train_Y = matdata[name2][0]
    test_X = matdata[name3]
    test_Y = matdata[name4][0]
    print("mat文件地址:", extract_name(address))
    print(" >训练集输入规模", train_X.shape, "训练集标签规模", train_Y.shape)
    print(" >测试集输入规模", test_X.shape, "测试集标签规模", test_Y.shape, "\n")
    return train_X, train_Y, test_X, test_Y


def saveTraintest(TraintestName, train_X, train_Y, test_X, test_Y, Validation_X=None, Validation_Y=None):
    if Validation_X is None:
        scipy.io.savemat(TraintestName, {'train_X': train_X, 'train_Y': train_Y, 'test_X': test_X, 'test_Y': test_Y})
        print("训练集测试集已经保存在：", extract_name(TraintestName), "\n")
    else:
        scipy.io.savemat(TraintestName,
                         {'train_X': train_X, 'train_Y': train_Y, 'test_X': test_X, 'test_Y': test_Y,
                          'Validation_X': Validation_X, 'Validation_Y': Validation_Y})
        print("训练集测试集已经保存在：", extract_name(TraintestName), "\n")
