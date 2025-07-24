from tkinter import *
from tkinter import messagebox, filedialog
import numpy as np
from scipy.io import loadmat
from src.Model_MainCode.accuracy import Accuracy
from src.Model_MainCode.MYAPI_sklearn.BPNN import BPNN
from src.gui.GetAddress import GetAddress


def get_path(address):
    """注意，以下列出的方法都是返回字符串而不是数据流"""
    # 返回一个字符串，且只能获取文件夹路径，不能获取文件的路径。
    path = filedialog.askdirectory(title='请选择一个目录')
    # 返回一个字符串，可以获取到任意文件的路径。
    # path = filedialog.askopenfilename(title='请选择文件')

    # 生成保存文件的对话框， 选择的是一个文件而不是一个文件夹，返回一个字符串。
    # path = filedialog.asksaveasfilename(title='请输入保存的路径')

    address.set(path)
    print(address.get())


def get_algorith(id):
    algorith_choosed.set(id)
    print("选择了算法", algorith_choosed.get())


def algorith(trainaddress, testaddress, algorithid, outPutText):
    print("start")
    # 读取文件
    outPutText.insert(INSERT, "正在读取文件中>>>")
    matdata = loadmat(trainaddress.get())  # 读取Mat文件, 文件里有训练集和测试集
    train_X = matdata['train_X']
    train_Y = matdata['train_Y'][0]
    matdata = loadmat(testaddress.get())  # 读取Mat文件, 文件里有训练集和测试集
    test_X = matdata['test_X']
    test_Y = matdata['test_Y'][0]
    n1 = np.size(train_X, 0)
    n2 = np.size(test_X, 0)
    m = np.size(train_X, 1)  # m=125
    outPutText.insert(INSERT, "\n训练集规模；" + str(n1) + "*" + str(m))
    outPutText.insert(INSERT, "\n训练集规模；" + str(n2) + "*" + str(m))

    # 选择算法
    # if algorithid==1:
    predict_Y = BPNN(train_X, train_Y, test_X, layersize=500, t=100)
    acc = Accuracy(predict_Y, test_Y, "all")

    # 输出准确率
    outPutText.insert(INSERT, "\n准确率为：")
    outPutText.insert(INSERT, acc)


if __name__ == '__main__':
    print("=====================log==========================")
    Tk = Tk()
    Tk.geometry("800x500+400+200")  # 定义窗口宽度，高度，左边距离，上面距离
    Tk.title("指令级侧信道逆向系统")

    # 训练集选择
    label01 = Label(Tk, text="训练数据地址")
    label01.place(x=10, y=10)

    trainaddress = StringVar()
    # trainaddress.set("D:/lc/5.科研/基于深度学习的安全芯片功耗侧信道逆向技术/数据及代码/SYS_CODE/src/static/DATA_100per")

    trainaddress.set("../DataFile/DataSets_Small/125d_train_test_0")
    entry01 = Entry(Tk, textvariable=trainaddress, font=('FangSong', 10), width=80, state='readonly')

    entry01.place(x=10, y=40)
    button01 = Button(Tk, text="选择路径", command=lambda: get_path(trainaddress))
    button01.place(x=600, y=40)

    # 测试集选择
    label02 = Label(Tk, text="测试数据地址")
    label02.place(x=10, y=70)

    testaddress = StringVar()
    testaddress.set("../DataFile/DataSets_Small/125d_train_test_0")
    entry02 = Entry(Tk, textvariable=testaddress, font=('FangSong', 10), width=80, state='readonly')
    entry02.place(x=10, y=100)
    button02 = Button(Tk, text="选择路径", command=lambda: get_path(testaddress))
    button02.place(x=600, y=100)

    # 算法选择
    label03 = Label(Tk, text="算法选择")
    label03.place(x=10, y=160)
    algorith_choosed = StringVar()
    algorithmbutton01 = Button(Tk, text="BPNN", activebackground="blue", command=lambda: get_algorith(1))
    algorithmbutton01.place(x=10, y=190)
    algorithmbutton02 = Button(Tk, text="CNN", command=lambda: get_algorith(2))
    algorithmbutton02.place(x=60, y=190)
    algorithmbutton03 = Button(Tk, text="KNN", command=lambda: get_algorith(3))
    algorithmbutton03.place(x=110, y=190)
    algorithmbutton04 = Button(Tk, text="随机森林", command=lambda: get_algorith(4))
    algorithmbutton04.place(x=160, y=190)

    # 模型计算
    startbutton = Button(Tk, text="开始运行",
                         command=lambda: algorith(trainaddress, testaddress, algorith_choosed, outPutText))
    startbutton.place(x=280, y=190)

    # 结果展示
    label03 = Label(Tk, text="运行结果")
    label03.place(x=10, y=230)

    outPutText = Text(Tk, height=15, width=100)
    outPutText.place(x=10, y=260)

    Tk.mainloop()  # 显示窗口
