"""
    @Author : ling chen
    @Create : 2023/03
    @Last modify: 2023/07
    @Description: 配置文件，定义了一些关于 DATA_3000per 数据集的全局常量，包括数据文件地址的存储等
"""
import os

# 获取当前脚本文件的绝对路径
Absolute_address = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")

# 相对路径
DIRECTORY_PREFIX = Absolute_address + "/DataFile/Datasets_Big/"

# 文件前缀名
FILENAME = ["/static/DATA_3000per/ADD A,#data/20220927-0003", "/static/DATA_3000per/ADDC A,Rn/20220927-0004",
            "/static/DATA_3000per/ANL A,Rn/20220927-0005", "/static/DATA_3000per/CLR A/20220927-0001",
            "/static/DATA_3000per/CPL A/20220927-0006", "/static/DATA_3000per/DA A/20220927-0007",
            "/static/DATA_3000per/DEC A/20220929-0001", "/static/DATA_3000per/DIV AB/20220929-0002",
            "/static/DATA_3000per/INC A/20220929-0003", "/static/DATA_3000per/JMP/20220929-0004",
            "/static/DATA_3000per/MOV Rn,A/20220927-0002", "/static/DATA_3000per/MUL AB/20221012-0001",
            "/static/DATA_3000per/NOP/20221012-0001", "/static/DATA_3000per/ORL A,Rn/20221012-0002",
            "/static/DATA_3000per/POP A/20221013-0001", "/static/DATA_3000per/PUSH A/20221013-0002",
            "/static/DATA_3000per/SETB C/20221013-0007", "/static/DATA_3000per/SUBB A,Rn/20221017-0001",
            "/static/DATA_3000per/SWAP A/20221017-0002", "/static/DATA_3000per/XCH A,Rn/20221017-0003",
            "/static/DATA_3000per/XRL A,Rn/20221017-0004"]

for i in range(len(FILENAME)):
    FILENAME[i] = Absolute_address + FILENAME[i]

# 指令字典：名称和对应数字标签
INSTRUCTION_DICT = {
    0: 'ADD A,Rn', 1: 'ADDC A,Rn', 2: 'ANL A,Rn', 3: 'CLR A', 4: 'CPL A', 5: 'DA A', 6: 'DEC A',
    7: 'DIV AB', 8: 'INC A', 9: 'JMP', 10: 'MOV Rn,A', 11: 'MUL AB', 12: 'NOP', 13: 'ORL A,Rn',
    14: 'POP A', 15: 'PUSH A', 16: 'SETB C', 17: 'SUBB A,Rn', 18: 'SWAP A', 19: 'XCH A,Rn', 20: 'XRL A,Rn'
}
# 提取键和值到两个列表
INSTRUCTION_LABEL = list(INSTRUCTION_DICT.keys())
INSTRUCTION_NAME = list(INSTRUCTION_DICT.values())

# 每条指令采集的样本数
SAMPLES_NUM = 500
