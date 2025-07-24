import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import pywt
import src.Configuration_matplot
from src.Configuration_05_4 import DIRECTORY_PREFIX
from src.Model_MainCode.FeatureEngineering.Component_DomainTransform import Component_DomainTransform
from src.Model_MainCode.Loadmatfile import loadData_stm32
from src.Model_MainCode.MYAPI_pytorch.__CNN_API__ import __load_cnnmodel__


def design_2d_filter(size, cutoff):
    """Design a 2D low-pass filter using sinc function and rectangular window."""
    p = np.linspace(-size // 2, size // 2, size)
    q = np.linspace(-size // 2, size // 2, size)
    P, Q = np.meshgrid(p, q)
    sinc_filter = np.sinc(2 * cutoff * P) * np.sinc(2 * cutoff * Q)
    window = np.outer(np.hanning(size), np.hanning(size))
    filter_coeffs = sinc_filter * window
    return filter_coeffs / np.sum(filter_coeffs)


def anti_mirror_filtering(X):
    anti_mirror_filter = design_2d_filter(31, 0.5)
    filtered_image = np.zeros_like(X)
    for idx in range(len(X)):
        image = X[idx]
        filtered_image[idx] = convolve2d(image, anti_mirror_filter, mode='same')

    return filtered_image


def anti_aliasing_filtering(X):
    anti_aliasing_filter = design_2d_filter(31, 0.25)

    filtered_image = np.zeros_like(X)
    for idx in range(len(X)):
        image = X[idx]
        filtered_image[idx] = convolve2d(image, anti_aliasing_filter, mode='same')

    return filtered_image


def Downsample(X):
    # Downsample the filtered image
    downsample_factor = 2
    downsampled_image = X[:, ::downsample_factor, :]

    print(f"Downsample: {X.shape}->{downsampled_image.shape}")
    return downsampled_image


def Upsample(X):
    # 上采样因子
    upsample_factor = 2
    original_shape = X.shape

    # 创建一个新的图像，大小增加了上采样因子的倍数
    upsampled_image = np.zeros((original_shape[0], original_shape[1] * upsample_factor, original_shape[2]))
    upsampled_image[:, ::upsample_factor, :] = X

    print(f"Upsample: {X.shape} -> {upsampled_image.shape}")
    return upsampled_image

# Generate a 1D test signal and perform CWT to get 2D representation
# np.random.seed(0)
# signal = np.random.rand(256)
X, Y_i, Y_d = loadData_stm32(DIRECTORY_PREFIX + 'DATA_(400d1000s).mat')
X = Component_DomainTransform(X[0:1000], 100)

X = Downsample(X)
X = anti_mirror_filtering(X)
# X = Upsample(X)
# X = anti_aliasing_filtering(X)
print(X.shape)
percentage, result = __load_cnnmodel__(X, "../Model_MainCode/0821_2002_cnn.pth")

acc = 0
for i in range(len(X)):
    print(result[i], " ", Y_i[i])
    if result[i] == Y_i[i]:
        acc += 1
print(acc / len(X))
# Apply anti-aliasing filtering


"""
Plotting results
"""
# plt.figure(figsize=(12, 8))
#
# plt.subplot(2, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')
# # plt.colorbar()
#
# plt.subplot(2, 2, 2)
# plt.imshow(filtered_image_mirror, cmap='gray')
# plt.title('抗镜像滤波后的图像')
# # plt.colorbar()
#
# plt.subplot(2, 2, 3)
# plt.imshow(downsampled_image, cmap='gray')
# plt.title('下采样图像')
# # plt.colorbar()
#
# plt.subplot(2, 2, 4)
# plt.imshow(filtered_image_aliasing, cmap='gray')
# plt.title('扛混叠滤波后的图像')
# # plt.colorbar()
#
# plt.tight_layout()
# plt.show()
