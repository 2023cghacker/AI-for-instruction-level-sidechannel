import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import pywt
import src.Configuration_matplot


def design_2d_filter(size, cutoff):
    """Design a 2D low-pass filter using sinc function and rectangular window."""
    p = np.linspace(-size // 2, size // 2, size)
    q = np.linspace(-size // 2, size // 2, size)
    P, Q = np.meshgrid(p, q)
    sinc_filter = np.sinc(2 * cutoff * P) * np.sinc(2 * cutoff * Q)
    window = np.outer(np.hanning(size), np.hanning(size))
    filter_coeffs = sinc_filter * window
    return filter_coeffs / np.sum(filter_coeffs)

# Generate a 1D test signal and perform CWT to get 2D representation
np.random.seed(0)
signal = np.random.rand(256)
scales = np.arange(1, 128)
coeffs, freqs = pywt.cwt(signal, scales, 'morl')
image = np.abs(coeffs)

# Design filters
anti_mirror_filter = design_2d_filter(31, 0.5)
anti_aliasing_filter = design_2d_filter(31, 0.25)

# Apply anti-mirror filtering
filtered_image_mirror = convolve2d(image, anti_mirror_filter, mode='same')

# Downsample the filtered image
downsample_factor = 2
downsampled_image = filtered_image_mirror[::downsample_factor, ::downsample_factor]

# Apply anti-aliasing filtering
filtered_image_aliasing = convolve2d(downsampled_image, anti_aliasing_filter, mode='same')

# Plotting results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.colorbar()
print("hello world")
plt.subplot(2, 2, 2)
plt.imshow(filtered_image_mirror, cmap='gray')
plt.title('抗镜像滤波后的图像')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(downsampled_image, cmap='gray')
plt.title('下采样图像')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(filtered_image_aliasing, cmap='gray')
plt.title('扛混叠滤波后的图像')
plt.colorbar()

plt.tight_layout()
plt.show()
