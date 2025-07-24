"""
    @Author : ling chen
    @Create : 2023/07
    @Last modify: 2023/07
    @Description: 数据配置文件入口，可以从这里选择 Configuration_n
"""
from src.Configuration_matplot import os

'''
使用不同配置时，import不同文件，注释掉其他的import
'''
# from src.Configuration_01 import DIRECTORY_PREFIX, FILENAME, INSTRUCTION_NAME, SAMPLES_NUM, Absolute_address, INSTRUCTION_LABEL
# from src.Configuration_02 import DIRECTORY_PREFIX, FILENAME, INSTRUCTION_NAME, SAMPLES_NUM, Absolute_address, INSTRUCTION_LABEL
#
# # from src.Configuration_03 import DIRECTORY_PREFIX, FILENAME, INSTRUCTION_NAME, SAMPLES_NUM, Absolute_address
from src.Configuration_03 import DIRECTORY_PREFIX, FILENAME, INSTRUCTION_NAME, INSTRUCTION_REGNAME, SAMPLES_NUM, \
    Absolute_address, INSTRUCTION_LABEL, INSTRUCTION_REGLABEL

# from src.Configuration_05 import DIRECTORY_PREFIX, FILENAME, INSTRUCTION_NAME, SAMPLES_NUM, Absolute_address
# 绝对路径
Absolute_address

# 相对路径
DIRECTORY_PREFIX

# 文件前缀名
FILENAME

# 指令名称
INSTRUCTION_NAME

# 寄存器名称
INSTRUCTION_REGNAME

# 指令标签1
INSTRUCTION_REGLABEL

# 指令标签2
INSTRUCTION_LABEL

# 每条指令采集的样本数
SAMPLES_NUM
