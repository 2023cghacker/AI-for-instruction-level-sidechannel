import scipy
from src.Configuration_05 import DIRECTORY_PREFIX
from src.Model_MainCode.FeatureEngineering.Component_DomainTransform import Component_DomainTransform
from src.Model_MainCode.Loadmatfile import loadData_stm32

X, Y_i, Y_d = loadData_stm32(DIRECTORY_PREFIX + 'DATA_(200d500s).mat')
print(X.shape, Y_i.shape, Y_d.shape)

for i in range(56):
    Xi = Component_DomainTransform(X[1000 * i:1000 * (i + 1), :], freqlen=50)  # 时频域转化组件
    print(Xi.shape)
    scipy.io.savemat(DIRECTORY_PREFIX + 'DATA_(50cwt500s)_{' + str(i) + '}.mat',
                     {'X': Xi, 'Y_i': Y_i[1000 * i:1000 * (i + 1)], 'Y_d': Y_d[1000 * i:1000 * (i + 1)]})
