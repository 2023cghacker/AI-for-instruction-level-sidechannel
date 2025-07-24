"""测试Entry组件的基本用法，使用面向对象的方式"""
from tkinter import *
from tkinter import messagebox


def login():
    print(1)
    print("用户名:" + rootTk.entry01.get())  # get()方法获取文本框的值
    print("密码:" + rootTk.entry02.get())
    messagebox.showinfo("学习系统", "登录成功！欢迎开始学习！")  # 弹窗


if __name__ == '__main__':
    rootTk = Tk()
    rootTk.geometry("800x500+400+200")  # 定义窗口高度，宽度，左边距离，上面距离

    """创建登录界面的组件"""
    # 1.用户名
    label01 = Label(rootTk, text="用户名")
    label01.pack()  # 显示
    '''
    # StringVar变量绑定到指定的组件
    # StringVar变量的值发生变化，组件内容也变化
    # 组件内容发生变化，StringVar变量的值也发生变化。
    '''
    v1 = StringVar()
    rootTk.entry01 = Entry(rootTk, textvariable=v1)
    rootTk.entry01.pack()
    v1.set("admin")
    print(v1.get())  # 获取变量值

    # 2.密码
    rootTk.label02 = Label(rootTk, text="密码")
    rootTk.label02.pack()

    v2 = StringVar()
    rootTk.entry02 = Entry(rootTk, textvariable=v2, show="*")
    rootTk.entry02.pack()

    # 3.登录按钮
    button1=Button(rootTk, text="登录", command=login)
    button1.place(x=400, y=95)

    rootTk.mainloop()  # 显示窗口
