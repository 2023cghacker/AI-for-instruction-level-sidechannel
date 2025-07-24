from scipy.io import loadmat
import numpy as np
from src.Configuration_01 import DIRECTORY_PREFIX, INSTRUCTION_NAME

add_pre = DIRECTORY_PREFIX
# matdata = loadmat(add_pre + 'instruction_pearsonr')  # 读取Mat文件
# pccs_matrix = matdata['pccs_matrix']
# print(pccs_matrix)  # 上三角矩阵
# for i in range(21):
#     print("和指令", i, "相似的有：", end="")
#     for j in range(21):
#         if pccs_matrix[i][j] > 0.3:
#             print(j, " ", end="")
#     print("\n")

'''
基于相似度进行分组，分组结果由KMEANS获得
'''
fname = '20d_(L)DATA.mat'
matdata = loadmat(add_pre + fname)  # 读取Mat文件
X = matdata['X']
Y = matdata['Y'][0]
new_Y = np.zeros((len(Y), 2))
type = [[0, 3, 8, 10], [7, 11, 14, 15, 18], [1, 5, 6, 16, 20], [2, 4, 9, 12, 13, 17, 19]]
for i in range(len(Y)):
    new_Y[i][0] = Y[i]
    if Y[i] in type[0]:
        new_Y[i][1] = 0
    elif Y[i] in type[1]:
        new_Y[i][1] = 1
    elif Y[i] in type[2]:
        new_Y[i][1] = 2
    elif Y[i] in type[3]:
        new_Y[i][1] = 3

insss = np.array(INSTRUCTION_NAME)
print(insss[type[0]])
print(insss[type[1]])
print(insss[type[2]])
print(insss[type[3]])
new_Y = np.trunc(new_Y).astype(int)
print(new_Y)

# filename = add_pre + "grouping_" + fname
# scipy.io.savemat(filename, {'X': X, 'Y': new_Y})
