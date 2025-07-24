# import scipy
# from scipy.io import loadmat
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import numpy as np
#
# # X = [[12, 4,3], [21, 4,1], [5, 12,7], [7, 2,12], [5, 1,6], [9, 5,8], [9, 0,2], [1, 12,1]]
# # Y = [1, 2, 3, 2, 2, 2, 3, 1]
# from src.Configuration_data_info import DIRECTORY_PREFIX
#
#
# def lda(X, Y, K):  # 降维到K维
#     lda = LinearDiscriminantAnalysis(n_components=K)
#     new_X = lda.fit_transform(X, Y)
#     transformation_matrix = lda.coef_
#     return new_X, transformation_matrix
#
#
# if __name__ == '__main__':
#     add_pre = DIRECTORY_PREFIX  # 目录地址前缀
#     matdata = loadmat(add_pre + '1Tl_train_test_0.mat')  # 读取Mat文件
#     train_X = matdata['train_X']
#     train_Y = matdata['train_Y'][0]
#     test_X = matdata['test_X']
#     test_Y = matdata['test_Y'][0]
#
#     k = 40
#     new_X, transformation_matrix = lda(train_X, train_Y, k)
#     test = test_X.dot(transformation_matrix.T)
#     print(transformation_matrix.shape)
#
#     # 将本次降维后的数据保存
#     savefilename = add_pre + str(k) + "d_(lda)train_test_0.mat"
#     scipy.io.savemat(savefilename, {'train_X': new_X, 'train_Y': train_Y, 'test_X': test_X, 'test_Y': test_Y})
import numpy as np


def lda(X, y, k):
    # 计算类别数
    classes = np.unique(y)
    C = len(classes)

    # 计算类别内部均值
    means = np.array([X[y == c].mean(axis=0) for c in classes])

    # 计算类别内部散布矩阵
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for c in classes:
        X_c = X[y == c]
        diff = X_c - means[c]
        Sw += np.dot(diff.T, diff)

    # 计算类别间散布矩阵
    overall_mean = X.mean(axis=0)
    Sb = np.zeros((X.shape[1], X.shape[1]))
    for c in classes:
        n_c = len(X[y == c])
        mean_diff = means[c] - overall_mean
        Sb += n_c * np.outer(mean_diff, mean_diff)

    # 计算广义特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

    # 选择前k个特征向量
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1][:k]]

    # 转换为新的特征空间
    X_lda = np.dot(X, eigenvectors)

    return X_lda, eigenvectors


if __name__ == '__main__':
    # 生成测试数据
    X = np.array([[1, 2, 5],
                  [2, 3, 6],
                  [3, 4, 3],
                  [4, 5, 2],
                  [4, 5, 8]])
    y = np.array([0, 2, 1, 1, 2])

    # 调用LDA算法
    k = 2
    X_lda, transform_matrix = lda(X, y, k)

    print("降维后的二维数组：")
    print(X_lda)
    print("转换矩阵：")
    print(transform_matrix)
