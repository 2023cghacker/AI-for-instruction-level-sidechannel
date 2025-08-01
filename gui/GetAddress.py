"""测试Entry组件的基本用法，使用面向对象的方式"""
from tkinter import *
from tkinter import messagebox, filedialog


def get_path(entry_text, address):
    """注意，以下列出的方法都是返回字符串而不是数据流"""
    # 返回一个字符串，且只能获取文件夹路径，不能获取文件的路径。
    path = filedialog.askdirectory(title='请选择一个目录')
    # 返回一个字符串，可以获取到任意文件的路径。
    # path = filedialog.askopenfilename(title='请选择文件')

    # 生成保存文件的对话框， 选择的是一个文件而不是一个文件夹，返回一个字符串。
    # path = filedialog.asksaveasfilename(title='请输入保存的路径')

    entry_text.set(path)
    address.set(path)
    print(address.get())


def GetAddress(address):
    GA_Tk = Tk()
    GA_Tk.title('选择')
    max_w, max_h = GA_Tk.maxsize()
    GA_Tk.geometry(f'500x300+{int((max_w - 500) / 2)}+{int((max_h - 300) / 2)}')  # 居中显示
    GA_Tk.resizable(width=False, height=False)

    # 标签组件
    label = Label(GA_Tk, text='选择目录：', font=('华文彩云', 15))
    label.place(x=50, y=100)

    # 输入框控件
    entry_text = StringVar()
    entry = Entry(GA_Tk, textvariable=entry_text, font=('FangSong', 10), width=30, state='readonly')
    entry.place(x=150, y=105)

    # 按钮控件

    button1 = Button(GA_Tk, text='选择路径', command=lambda: get_path(entry_text, address))
    button1.place(x=400, y=95)

    GA_Tk.mainloop()
    print("close")


# GetAddress()
