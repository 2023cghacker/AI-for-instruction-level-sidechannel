import numpy as np  # 使用数组
import matplotlib.pyplot as plt  # 可视化
from matplotlib import rcParams  # 图大小
import src.Configuration_matplot
from scipy.io import loadmat
from sklearn.tree import DecisionTreeClassifier as dtc  # 树算法
from sklearn.tree import plot_tree  # 树图
from src.Configuration_03 import DIRECTORY_PREFIX, INSTRUCTION_NAME, INSTRUCTION_LABEL
from src.Model_MainCode.accuracy import Accuracy

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

rcParams['figure.figsize'] = (50, 40)

# 导入训练集和测试集
address = DIRECTORY_PREFIX + "1T_traintest_m(500d100s).mat"
# address = "train_test_0"
matdata = loadmat(address)  # 读取Mat文件, 文件里有训练集和测试集
train_X = matdata['train_X']
train_Y = matdata['train_Y2'][0]
test_X = matdata['test_X']
test_Y = matdata['test_Y2'][0]
n1 = np.size(train_X, 0)
n2 = np.size(test_X, 0)
m = np.size(train_X, 1)  # m=125
print("训练集规模；", n1, "*", m)
print("测试集规模；", n2, "*", m)

model = dtc(criterion='entropy', max_depth=100)
model.fit(train_X, train_Y)

predict_Y = model.predict(test_X)
print("预测结果为：")
print(predict_Y)

Accuracy(predict_Y, test_Y, 2, INSTRUCTION_NAME, INSTRUCTION_LABEL)
# print(cl('Accuracy of the model is {:.0%}'.format(accuracy_score(y_test, pred_model)), attrs=['bold']))

# feature_names = df.columns[:5]
# target_names = df['Drug'].unique().tolist()

# feature_names = ['层数,基底面积', '总面积', '院落面积', '院落面积和建筑基底面积比值',
#                  '屋顶坡度', '屋面角度', '山墙面是否悬挑', '屋顶弧度',
#                  '正房主房披檐处临院落高度', '是否露台', '作为露台的屋顶面积', '正房临露台处高度',
#                  '屋顶一共几面坡', '屋顶露台面积和基底面积的比值']
# target_names = ['丽江', '金塘', '晋宁', '腾冲五合', '石屏杨新寨']
plot_tree(model,
          # feature_names=feature_names,
          # class_names=target_names,
          filled=True,
          rounded=True)

plt.savefig('tree_visualization.png')
