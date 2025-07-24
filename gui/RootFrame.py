"""测试Entry组件的基本用法，使用面向对象的方式"""
from tkinter import *
from tkinter import messagebox
from src.gui.GetAddress import GetAddress


def ChooseAddress(address):
    print("ChooseAddress")
    GetAddress(address)


if __name__ == '__main__':
    print("=====================log==========================")
    Tk = Tk()
    Tk.geometry("800x500+400+200")  # 定义窗口高度，宽度，左边距离，上面距离
    Tk.title("指令级侧信道逆向系统")

    # 训练集选择
    label01 = Label(Tk, text="训练数据地址", bg="blue")
    label01.place(x=10, y=10)

    trainaddress = StringVar()
    trainaddress.set("D:/lc/5.科研/基于深度学习的安全芯片功耗侧信道逆向技术/数据及代码/SYS_CODE/src/static/DATA_100per")
    button01 = Button(Tk, textvariable=trainaddress, command=lambda: ChooseAddress(trainaddress))
    button01.place(x=10, y=40)

    # 测试集选择
    label02 = Label(Tk, text="测试数据地址", bg="blue")
    label02.place(x=10, y=80)

    testaddress = StringVar()
    testaddress.set("D:/lc/5.科研/基于深度学习的安全芯片功耗侧信道逆向技术/数据及代码/SYS_CODE/src/static/DATA_100per")
    button02 = Button(Tk, textvariable=testaddress, command=lambda: ChooseAddress(testaddress))
    button02.place(x=10, y=110)

    # 算法选择
    label03 = Label(Tk, text="算法选择", bg="blue")
    label03.place(x=10, y=80)

    testaddress = StringVar()
    testaddress.set("D:/lc/5.科研/基于深度学习的安全芯片功耗侧信道逆向技术/数据及代码/SYS_CODE/src/static/DATA_100per")
    button02 = Button(Tk, textvariable=testaddress, command=lambda: ChooseAddress(testaddress))
    button02.place(x=10, y=110)
    Tk.mainloop()  # 显示窗口
