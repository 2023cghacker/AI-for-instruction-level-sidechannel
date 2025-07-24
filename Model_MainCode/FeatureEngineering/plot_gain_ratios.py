import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_gain_ratios(gain_ratios, top_K_indices, data_shape):
    if data_shape is None:
        # 一维时序数据的绘图
        plt.figure(figsize=(15, 8))
        plt.plot(gain_ratios, label="Gain Ratios")
        plt.scatter(top_K_indices, gain_ratios[top_K_indices], color="red", label="Top K Points")
        plt.xlabel("Feature Index")
        plt.ylabel("Gain Ratio")
        plt.title("Gain Ratios and Top K Points(K=" + str(len(top_K_indices)) + ")")
        plt.legend()
        plt.show()

    else:
        # 二维时频域数据的绘图
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, projection='3d')
        reshaped_ratios = gain_ratios.reshape(data_shape)

        X = np.arange(data_shape[1])
        Y = np.arange(data_shape[0])
        X, Y = np.meshgrid(X, Y)
        Z = reshaped_ratios

        #       ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.plot_wireframe(X, Y, Z, cmap='viridis', linewidth=0.2)

        # 标注比值最大的K个点
        top_K_coords = np.unravel_index(top_K_indices, data_shape)
        ax.scatter(top_K_coords[1], top_K_coords[0], Z[top_K_coords], color='red', s=2)

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Gain Ratio')
        ax.set_title('Gain Ratios Surface and Top K Points')
        plt.show()
