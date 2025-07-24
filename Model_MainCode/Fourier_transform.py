import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import scipy
from scipy.io import loadmat
from src.Configuration_01 import DIRECTORY_PREFIX

add_pre = DIRECTORY_PREFIX
matdata = loadmat(add_pre + '125d_DATA')  # 读取Mat文件, 文件里有训练集和测试集
X = matdata['X']
Y = matdata['Y']
n = np.size(X, 0)
m = np.size(X, 1)
t = np.array(np.linspace(0, m, m))  # 时间[0, m]内均匀取m个点

'''
将所有功率片段从时域转成频域，频域长度也是125
'''
lenth = n  # 选多少个样本参与计算
pows = np.zeros([lenth, m])
for i in range(0, lenth):
    # 傅里叶变换
    pow = fft.fft(X[i])
    pows[i] = pow
    freqs = fft.fftfreq(t.size, t[1] - t[0])

'''
对频域进行方差统计，得到一个1*125的向量
向量反应了每个频率值上的方差，也就是信息量，越高的峰值表示该点拥有越高的熵。
'''
print("频域矩阵的大小为：", pows.shape)
pows_var = np.zeros(m)
for i in range(m):
    a = pows[:, i]
    pows_var[i] = np.var(a)

plt.plot(freqs[freqs > 0], pows_var[freqs > 0])
plt.title("频域上的方差/熵")
plt.show()

# N = 0
# pows_Difference = np.zeros(m)
# for i in range(lenth - 1):
#     print(i)
#     a = np.abs(pows[lenth - 1] - pows[i])
#     pows_Difference += a
#     N = N + 1
#
# pows_Difference = pows_Difference / N
# print(pows_Difference)
# plt.plot(freqs[freqs > 0], pows_Difference[freqs > 0])
# plt.title("频域上的方差/熵")
# plt.show()

'''
保留高方差的频域坐标点，其余的视作噪声进行清除
'''
index = np.where(pows_var > 0.0001)
print(index[0])
print("高信息点的个数为：", len(index[0]))
plt.plot(freqs, pows_var)
plt.plot(freqs[index], pows_var[index], '*')
plt.title("高信息点")
plt.show()

'''
对每一个样本进行频域去噪，然后傅里叶逆变换得到去噪后的时域图像。
'''
index = np.where(pows_var < 0.0001)  # 低方差/低熵
plot_flag = 1

new_X = np.zeros([n, m])
for i in range(0, n):
    Highentropy_pows = pows[i].copy()
    Highentropy_pows[index] = 0
    # 傅里叶逆变换
    filter_sign = fft.ifft(Highentropy_pows).real
    new_X[i] = filter_sign

    if plot_flag==1:
        plt.subplot(221)
           

        plt.subplot(222)
        plt.plot(freqs[freqs > 0], pows[i][freqs > 0], c='orangered', label='Frequency')
        plt.title('频域图')
        plt.xlabel('Frequency 频率')
        plt.ylabel('Power 功率')
        plt.tick_params(labelsize=10)
        plt.grid(linestyle=':')
        plt.tight_layout()

        plt.subplot(223)
        plt.plot(freqs[freqs > 0], Highentropy_pows[freqs > 0], c='orangered', label='Frequency')
        plt.title('去噪后的频域图')
        plt.xlabel('Frequency 频率')
        plt.ylabel('Power 功率')
        plt.tick_params(labelsize=10)
        plt.grid(linestyle=':')
        plt.tight_layout()

        plt.subplot(224)
        plt.plot(t, filter_sign)
        plt.title("去噪后的时域图", fontsize=12)
        plt.xlabel('T 时间')
        plt.ylabel('Power 功率')
        plt.grid(linestyle=":")
        plt.show()

print(new_X.shape)
scipy.io.savemat(add_pre + "Fourier_125d_DATA.mat", {'X': new_X, 'Y': Y})
