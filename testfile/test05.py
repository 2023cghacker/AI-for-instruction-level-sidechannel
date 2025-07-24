import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以便结果可复现
np.random.seed(0)

# 生成一个50x50的基础图像
base_image = np.random.rand(50, 50)  # 使用0到1之间的随机浮点数填充

# 创建第二个图像，它将是基础图像加上一些小的随机噪声
noise = 0.05 * np.random.randn(50, 50)  # 生成与基础图像同样大小的噪声图像，噪声幅度为0.05
similar_image = base_image + noise

# 确保像素值在0到1之间
similar_image = np.clip(similar_image, 0, 1)

# 计算MSE
mse = np.mean((base_image - similar_image) ** 2)

# 输出MSE值
print(f"MSE: {mse}")

# 可视化图像（可选）
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(base_image, cmap='gray')
plt.title('Base Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(similar_image, cmap='gray')
plt.title('Similar Image')
plt.axis('off')

plt.show()