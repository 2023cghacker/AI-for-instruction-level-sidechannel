import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.signal import convolve2d
import pywt
import src.Configuration_matplot
from src.Configuration_05_4 import DIRECTORY_PREFIX
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
    #扛镜像，针对上采样
    anti_mirror_filter = design_2d_filter(31, 0.5)
    filtered_image = np.zeros_like(X)
    for idx in range(len(X)):
        image = X[idx]
        filtered_image[idx] = convolve2d(image, anti_mirror_filter, mode='same')

    return filtered_image


def anti_aliasing_filtering(X):
    #扛混叠，针对下采样
    anti_aliasing_filter = design_2d_filter(31, 0.25)

    filtered_image = np.zeros_like(X)
    for idx in range(len(X)):
        image = X[idx]
        filtered_image = convolve2d(image, anti_aliasing_filter, mode='same')

    return filtered_image


def Downsample(X):
    # Downsample the filtered image
    downsample_factor = 2
    downsampled_image = X[:, :, ::downsample_factor]

    print(f"Downsample: {X.shape}->{downsampled_image.shape}")
    return downsampled_image


def Upsample(X):
    # 获取输入图像的尺寸
    height, width = X.shape[1], X.shape[2]

    # 计算新的尺寸（上采样因子为2）
    upsample_factor = 2
    # new_height = height * upsample_factor
    new_width = width * upsample_factor

    # 初始化一个全零的新图像
    upsampled_image = np.zeros((X.shape[0], height, new_width), dtype=X.dtype)

    # 将原图像的值复制到新图像的正确位置
    upsampled_image[:, :, ::upsample_factor, ] = X

    print(f"Upsample: {X.shape} -> {upsampled_image.shape}")
    return upsampled_image

X, Y_i, Y_d = loadData_stm32(DIRECTORY_PREFIX + 'DATA_(100d1000s).mat')
signal = X[0]
scales = np.arange(1, 101)
coeffs, freqs = pywt.cwt(signal, scales, 'morl')
image = np.abs(coeffs)
print(image.shape)
image = image.reshape(100, 100)

font = FontProperties(family='Times New Roman', size=14, weight='bold')
us_image = Upsample(image.reshape(1, 100, 100))
plt.subplot(2, 2, 1)
plt.imshow(us_image.reshape(100, 200), aspect='auto')
plt.title('Upsample Image',font=font)
plt.axis('off')  # 去掉坐标轴

f_image = anti_aliasing_filtering(us_image.reshape(1, 100, 200))
plt.subplot(2, 2, 2)
plt.imshow(f_image.reshape(100, 200), aspect='auto')
plt.title('Filtered Image',font=font)
plt.axis('off')  # 去掉坐标轴

X, Y_i, Y_d = loadData_stm32(DIRECTORY_PREFIX + 'DATA_(400d1000s).mat')
signal = X[0]
scales = np.arange(1, 101)
coeffs, freqs = pywt.cwt(signal, scales, 'morl')
image = np.abs(coeffs)
print(image.shape)
image = image.reshape(100, 400)

ds_image = Downsample(image.reshape(1, 100, 400))
plt.subplot(2, 2, 3)
plt.imshow(ds_image.reshape(100, 200), aspect='auto')
plt.title('Downsample Image',font=font)
plt.axis('off')  # 去掉坐标轴

f_image = anti_aliasing_filtering(ds_image.reshape(1, 100, 200))
plt.subplot(2, 2, 4)
plt.imshow(f_image.reshape(100, 200), aspect='auto')
plt.title('Filtered Image',font=font)
plt.axis('off')  # 去掉坐标轴

plt.tight_layout()
plt.show()
