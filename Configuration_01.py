"""
    @Author : ling chen
    @Create : 2023/03
    @Last modify: 2023/07
    @Description: 配置文件，定义了一些关于 DATA_100per 数据集的全局常量，包括数据文件地址的存储等
"""
import os

# 获取当前脚本文件的绝对路径
Absolute_address = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")

# 相对路径
DIRECTORY_PREFIX = Absolute_address + "/DataFile/DataSets_Small/"

# 文件前缀名
FILENAME = ["/static/DATA_100per/ADD/20220409-0003", "/static/DATA_100per/ADDC/20220409-0021",
            "/static/DATA_100per/ANL/20220409-0012", "/static/DATA_100per/CLR/20220409-0008",
            "/static/DATA_100per/CPL/20220409-0016", "/static/DATA_100per/DA/20220409-0023",
            "/static/DATA_100per/DEC/20220409-0005", "/static/DATA_100per/DIV/20220409-0026",
            "/static/DATA_100per/INC/20220409-0004", "/static/DATA_100per/JMP/20220409-0007",
            "/static/DATA_100per/MOV/20220409-0002", "/static/DATA_100per/MUL/20220409-0025",
            "/static/DATA_100per/NOP/20220409-0024", "/static/DATA_100per/ORL/20220409-0013",
            "/static/DATA_100per/POP/20220409-0010", "/static/DATA_100per/PUSH/20220409-0009",
            "/static/DATA_100per/SETB/20220409-0015", "/static/DATA_100per/SUBB/20220409-0022",
            "/static/DATA_100per/SWAP/20220409-0006", "/static/DATA_100per/XCH/20220409-0011",
            "/static/DATA_100per/XRL/20220409-0014"]
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
SAMPLES_NUM = 100
