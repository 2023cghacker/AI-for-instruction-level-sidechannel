"""
    @Author : ling chen
    @Create : 2024/7
    @Last modify: 2024/7
    @Description: 配置文件，定义了一些关于 trace_crossresistance 数据集的全局常量，包括数据文件地址的存储等
"""

import os

# 获取当前脚本文件的绝对路径
Absolute_address = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")

# 相对路径
DIRECTORY_PREFIX = Absolute_address + "/DataFile/DataSets_stm32/"

# 文件前缀名
FILENAME = ["/static/stm32_DATA_1000per/device9_400d/AND,ORR,EOR,BIC/20240731_",
            "/static/stm32_DATA_1000per/device9_400d/CMP,CMN,TST,TEQ/20240731_",
            "/static/stm32_DATA_1000per/device9_400d/LSL,LSR,ASR,LDR/20240731_",
            "/static/stm32_DATA_1000per/device9_400d/MOV,ADD,SUB,MUL/20240731_",]

for i in range(len(FILENAME)):
    FILENAME[i] = Absolute_address + FILENAME[i]

# 指令字典：名称和对应数字标签
INSTRUCTION_DICT = {
    0: 'AND', 1: 'ORR', 2: 'EOR', 3: 'BIC', 4: 'CMP', 5: 'CMN',
    6: 'TST', 7: 'TEQ', 8: 'LSL', 9: 'LSR', 10: 'ASR',
    11: 'LDR', 12: 'MOV', 13: 'ADD', 14: 'SUB', 15: 'MUL',
    # 16: 'STR', 17: 'PUSH', 18: 'POP', 19: 'NOP',
}
# 提取键和值到两个列表
INSTRUCTION_LABEL = list(INSTRUCTION_DICT.keys())
INSTRUCTION_NAME = list(INSTRUCTION_DICT.values())

# 每条指令采集的样本数
SAMPLES_NUM = 1000
