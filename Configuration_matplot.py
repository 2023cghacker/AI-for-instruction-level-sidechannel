"""
    @Author : ling chen
    @Create : 2023/03
    @Last modify: 2023/07
    @Description: matplot的一些配置信息 中文绘图配置
"""

import matplotlib.pyplot as plt
import os
import PySide2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
