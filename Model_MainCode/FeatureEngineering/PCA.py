import numpy as np
import scipy
from scipy.io import loadmat
from src.Configuration_04 import DIRECTORY_PREFIX


# 定义PCA降维函数
def pca(data_mat, N):
    # 求平均值
    mean_val = np.mean(data_mat, axis=0)
    # print("求平均值:\n", mean_val)

    # 去中心化
    mean_removed = data_mat - mean_val
    # print("去中心化:\n", mean_removed)

    # 获取协方差矩阵
    cov_mat = np.cov(mean_removed, rowvar=0)
    # print("协方差矩阵:\n", cov_mat)

    #  获取特征根及特征向量
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    total = sum(eigen_vals)  # 总方差
    # print("特征根及特征向量:\n", eigen_vals, "\n", eigen_vecs)

    # 特征根排序
    eigen_val_ind = np.argsort(eigen_vals)
    # print("特征根排序:\n", eigen_val_ind)

    # 删除解释量小的特征根
    eigen_val_ind = eigen_val_ind[:-(N + 1):-1]
    ratio = sum(eigen_vals[eigen_val_ind]) / total * 100
    # print("删除解释量小后的特征根排序:\n", eigen_val_ind,"\n方差占比：", ratio, '%')

    # 由高到低排序
    red_eigen_vecs = eigen_vecs[:, eigen_val_ind]
    # print("由高到低排序:\n", red_eigen_vecs)

    # l新维度的数据
    low_data_mat = np.dot(mean_removed, red_eigen_vecs)
    # low_data_mat = mean_removed * red_eigen_vecs
    # print("新维度的数据:\n", low_data_mat)

    #  获取目标向量值
    recon_mat = np.dot(low_data_mat, red_eigen_vecs.T) + mean_val
    # print("获取目标向量值:\n", recon_mat)

    # 返回新维度(N维)的数据,特征向量,贡献度
    return low_data_mat, red_eigen_vecs, ratio


if __name__ == '__main__':
    matdata = loadmat(DIRECTORY_PREFIX + 'DATA_m(750d200s).mat')  # 读取Mat文件
    X = matdata['X']
    Y = matdata['Y'][0]
    print(">原数据集规模：", X.shape)
    print(X)

    '''
        PCA降成k维
    '''
    k = 40
    new_X, eigen_vecs, ratio = pca(X, k)  # 返回降维后新的数据集，特征向量，贡献度

    print("========================================================\n")
    print(k, "维数据集信息量占比", ratio, "%,  新数据集规模：", new_X.shape)
    # print(new_X)
    print("\n========================================================\n")

    # 将本次降维后的数据保存
    filename = DIRECTORY_PREFIX + str(k) + "d_(pca)DATA.mat"
    print(filename)
    # scipy.io.savemat(filename, {'X': new_X, 'Y': Y, 'eigen_vecs': eigen_vecs})
