import numpy as np
import pywt
import matplotlib.pyplot as plt


def cwt_transform(signal, x, y, wavelet='cmor'):
    # Calculate the Continuous Wavelet Transform
    scales = np.linspace(1, len(signal), y)
    coef, freqs = pywt.cwt(signal, scales, wavelet)
    print(coef.shape)
    # Resize the result to the desired dimensions
    time_resampled = np.linspace(0, len(signal) - 1, x)
    freq_resampled = np.linspace(0, len(signal) - 1, y)

    coef_resampled = np.zeros((y, x))
    for i in range(y):
        coef_resampled[i, :] = np.interp(time_resampled, np.arange(len(signal)), coef[i, :])

    print(f"原信号维度={signal.shape},时频域转换后信号维度={coef_resampled.shape}")
    return coef_resampled


# Example usage
if __name__ == "__main__":
    # Generate a sample signal
    t = np.linspace(0, 1, 200, endpoint=False)
    signal = np.cos(2 * np.pi * 7 * t) + np.sin(2 * np.pi * 13 * t)

    # Desired dimensions for the output
    x = 100  # Time dimension
    y = 100  # Frequency dimension

    plt.plot(t, signal)
    plt.show()
    # Perform CWT and get the transformed signal
    transformed_signal = cwt_transform(signal, x, y)

    # Plot the result
    plt.imshow(transformed_signal, extent=[0, 1, 1, 0], aspect='auto', cmap='jet')
    plt.title('Continuous Wavelet Transform')
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.colorbar(label='Magnitude')
    plt.show()
