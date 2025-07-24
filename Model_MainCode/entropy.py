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