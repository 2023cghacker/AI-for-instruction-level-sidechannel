import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from scipy.io import loadmat
import tensorflow


def myLSTM(train_X, train_Y, test_X, test_Y):
    n1, n2, m = np.size(train_X, 0), np.size(test_X, 0), np.size(train_X, 1)

    # 构建 LSTM 模型
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, m)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 将数据转换为模型可接受的形式
    train_X, train_Y = train_X.reshape(n1, 1, m), train_Y.reshape(n1, 1)
    test_X, test_Y = test_X.reshape(n2, 1, m), test_Y.reshape(n2, 1)
    '''
    参数意义：
    （1）batch_size
    单次训练用的样本数，通常为2^N，如32、64、128…
    例如：batch_size=32为每一次载入数据集为32个样本数。
    相对于正常数据集，如果过小，训练数据就收敛困难；过大，虽然相对处理速度加快，但所需内存容量增加。
    要根据电脑的内存容量进行设定，使用中需要根据计算机性能和训练次数之间平衡。
    另：如果batch_size设置为最大，就是原始的梯度下降。
    如果batch_size设置为1，则位随机梯度下降。
    （2）iteration
    1个iteration等于使用batch_size个的样本训练一次；
    例如：iteration=2为每一次batch_size=32载入数据时，迭代训练2次。
    （3）epochs
    1个epoch等于使用训练集中的全部样本训练一次；
    例如:epoch=4为全部样本训练4次。
    '''

    # 训练 LSTM 模型
    model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=2)

    # 使用 LSTM 模型进行预测
    predictions = model.predict(test_X)

    # 计算预测误差
    error = np.sqrt(np.mean((predictions - test_X) ** 2))

    # 可视化结果
    plt.plot(test_Y, label='Test_Y', )
    plt.plot(predictions, color='red', label='Predictions')
    plt.legend(loc='upper left')
    plt.show()

    return predictions
