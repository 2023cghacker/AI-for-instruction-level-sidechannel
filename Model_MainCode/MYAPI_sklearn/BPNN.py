import time
import joblib
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def CreateBPNN(train_X, train_Y, test_X, layersize, t, saveflag=False):
    # 数据标准化：神经网络对数据尺度敏感，所以最好在训练前标准化，或者归一化，或者缩放到[-1,1]
    scaler = StandardScaler()  # 标准化转换
    scaler.fit(test_X)  # 训练标准化对象
    x_test_Standard = scaler.transform(test_X)  # 转换数据集
    scaler.fit(train_X)  # 训练标准化对象
    x_train_Standard = scaler.transform(train_X)  # 转换数据集

    # 开始训练
    bp = MLPClassifier(hidden_layer_sizes=(layersize,), activation='relu',
                       solver='adam', alpha=0.0001, batch_size='auto',
                       learning_rate='invscaling', max_iter=t)

    bp.fit(x_train_Standard, train_Y.astype('int'))
    predict_Y = bp.predict(x_test_Standard)

    # print( classification_report(y_test.astype('int'), y_predict))
    if saveflag:
        # 保存bp神经网络和训练参数
        date = time.strftime('%m%d_%H%M', time.localtime())  # %Y年份，M月份以此类推
        model_name = date + "_bpnn.m"
        joblib.dump(bp, model_name)
        print("神经网络保存在", model_name, "文件中")

    return predict_Y


def LoadBPNN(model_path, test_X):
    # 从已有的bp神经网络中导入
    bpnn = joblib.load(model_path)
    scaler = StandardScaler()  # 标准化转换
    scaler.fit(test_X)  # 训练标准化对象
    x_test_Standard = scaler.transform(test_X)  # 转换数据集
    predict_Y = bpnn.predict(x_test_Standard)

    return predict_Y
