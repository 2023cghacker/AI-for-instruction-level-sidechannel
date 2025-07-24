import scipy
import numpy as np
# from src.Model_MainCode.FeatureEngineering.Interpolation import DimensionalityReduction
from src.Model_MainCode.FeatureEngineering.LDA import lda
from src.Model_MainCode.FeatureEngineering.PCA import pca


def Component_Reduction(trainX, trainY, testX, testY, methond, newD, savepath=None):
    print("befor Dimension reduction: 数据集维度：", trainX.shape)

    k = newD
    if methond == "PCA":
        new_trainX, transformation_matrix, ratio = pca(trainX, k)  # 返回降维后新的数据集，特征向量，贡献度
        print("after Dimension reduction PCA: 新数据集维度", new_trainX.shape, "信息量占比", ratio, "%")
        new_testX = np.dot(testX, transformation_matrix)
    elif methond == "LDA":
        new_trainX, transformation_matrix = lda(trainX, trainY, k)  # 返回降维后新的数据集，特征向量，贡献度
        print("after Dimension reduction LDA: 新数据集维度", new_trainX.shape)
        new_testX = np.dot(testX, transformation_matrix)
    else:
        print("降维参数输入错误")

    # elif methond == "linspace":
    #     new_X = DimensionalityReduction(X, k)
    #     print("after Dimension reduction linspace: ", "新数据集规模：", new_X.shape)

    if savepath is not None:
        print("降维后的数据文件已保存至", savepath)
        # scipy.io.savemat(savepath, {'train_X': new_trainX, 'Y': trainY, 'eigen_vecs': eigen_vecs})

    return new_trainX, trainY, new_testX, testY
