"""
    @Author : ling chen
    @Create : 2023/07
    @Last modify: 2023/07
    @Description: 配置文件，定义了一些关于 registerbasedtrace 数据集的全局常量，包括数据文件地址的存储等
"""

import os

# 获取当前脚本文件的绝对路径
Absolute_address = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")

# 相对路径
DIRECTORY_PREFIX = Absolute_address + "/DataFile/DataSets_registerbased/"

# 文件前缀名
FILENAME = ["/static/registerbasedtrace/ADD A, Rn/ADD A, R1/20230313-0001",
            "/static/registerbasedtrace/ADD A, Rn/ADD A, R2/20230313-0002",
            "/static/registerbasedtrace/ADD A, Rn/ADD A, R3/20230313-0003",
            "/static/registerbasedtrace/ADD A, Rn/ADD A, R4/20230313-0004",
            "/static/registerbasedtrace/ADD A, Rn/ADD A, R5/20230313-0005",
            "/static/registerbasedtrace/ADD A, Rn/ADD A, R6/20230313-0006",
            "/static/registerbasedtrace/ADD A, Rn/ADD A, R7/20230313-0007",
            "/static/registerbasedtrace/ADDC A, Rn/ADDC A, R0/20230313-0002",
            "/static/registerbasedtrace/ADDC A, Rn/ADDC A, R1/20230313-0008",
            "/static/registerbasedtrace/ADDC A, Rn/ADDC A, R2/20230313-0009",
            "/static/registerbasedtrace/ADDC A, Rn/ADDC A, R3/20230313-0010",
            "/static/registerbasedtrace/ADDC A, Rn/ADDC A, R4/20230313-0011",
            "/static/registerbasedtrace/ADDC A, Rn/ADDC A, R5/20230313-0012",
            "/static/registerbasedtrace/ADDC A, Rn/ADDC A, R6/20230313-0013",
            "/static/registerbasedtrace/ADDC A, Rn/ADDC A, R7/20230313-0014",
            "/static/registerbasedtrace/ANL A, Rn/ANL A, R0/20230313-0015",
            "/static/registerbasedtrace/ANL A, Rn/ANL A, R1/20230313-0016",
            "/static/registerbasedtrace/ANL A, Rn/ANL A, R2/20230313-0017",
            "/static/registerbasedtrace/ANL A, Rn/ANL A, R3/20230313-0018",
            "/static/registerbasedtrace/ANL A, Rn/ANL A, R4/20230313-0019",
            "/static/registerbasedtrace/ANL A, Rn/ANL A, R5/20230313-0020",
            "/static/registerbasedtrace/ANL A, Rn/ANL A, R6/20230313-0021",
            "/static/registerbasedtrace/ANL A, Rn/ANL A, R7/20230313-0022",
            "/static/registerbasedtrace/DEC Rn/DEC R0/20230314-0001",
            "/static/registerbasedtrace/DEC Rn/DEC R1/20230314-0002",
            "/static/registerbasedtrace/DEC Rn/DEC R2/20230314-0003",
            "/static/registerbasedtrace/DEC Rn/DEC R3/20230314-0004",
            "/static/registerbasedtrace/DEC Rn/DEC R4/20230314-0005",
            "/static/registerbasedtrace/DEC Rn/DEC R5/20230314-0006",
            "/static/registerbasedtrace/DEC Rn/DEC R6/20230314-0007",
            "/static/registerbasedtrace/DEC Rn/DEC R7/20230314-0008",
            "/static/registerbasedtrace/DJNZ Rn, rel/DJNZ R0, rel/20230403-0018",
            "/static/registerbasedtrace/DJNZ Rn, rel/DJNZ R1, rel/20230406-0001",
            "/static/registerbasedtrace/DJNZ Rn, rel/DJNZ R2, rel/20230406-0002",
            "/static/registerbasedtrace/DJNZ Rn, rel/DJNZ R3, rel/20230406-0003",
            "/static/registerbasedtrace/DJNZ Rn, rel/DJNZ R4, rel/20230406-0004",
            "/static/registerbasedtrace/DJNZ Rn, rel/DJNZ R5, rel/20230406-0005",
            "/static/registerbasedtrace/DJNZ Rn, rel/DJNZ R6, rel/20230406-0006",
            "/static/registerbasedtrace/DJNZ Rn, rel/DJNZ R7, rel/20230406-0007",
            "/static/registerbasedtrace/INC Rn/INC R0/20230314-0009",
            "/static/registerbasedtrace/INC Rn/INC R1/20230314-0010",
            "/static/registerbasedtrace/INC Rn/INC R2/20230314-0011",
            "/static/registerbasedtrace/INC Rn/INC R3/20230314-0012",
            "/static/registerbasedtrace/INC Rn/INC R4/20230314-0013",
            "/static/registerbasedtrace/INC Rn/INC R5/20230314-0014",
            "/static/registerbasedtrace/INC Rn/INC R6/20230314-0015",
            "/static/registerbasedtrace/INC Rn/INC R7/20230314-0016",
            "/static/registerbasedtrace/MOV A, Rn/MOV A, R0/20230314-0017",
            "/static/registerbasedtrace/MOV A, Rn/MOV A, R1/20230314-0018",
            "/static/registerbasedtrace/MOV A, Rn/MOV A, R2/20230314-0019",
            "/static/registerbasedtrace/MOV A, Rn/MOV A, R3/20230314-0020",
            "/static/registerbasedtrace/MOV A, Rn/MOV A, R4/20230314-0021",
            "/static/registerbasedtrace/MOV A, Rn/MOV A, R5/20230314-0001",
            "/static/registerbasedtrace/MOV A, Rn/MOV A, R6/20230314-0001",
            "/static/registerbasedtrace/MOV A, Rn/MOV A, R7/20230314-0002",
            "/static/registerbasedtrace/MOV Rn, #data/MOV R0, #data/20230403-0001",
            "/static/registerbasedtrace/MOV Rn, #data/MOV R1, #data/20230403-0003",
            "/static/registerbasedtrace/MOV Rn, #data/MOV R2, #data/20230403-0004",
            "/static/registerbasedtrace/MOV Rn, #data/MOV R3, #data/20230403-0005",
            "/static/registerbasedtrace/MOV Rn, #data/MOV R4, #data/20230403-0006",
            "/static/registerbasedtrace/MOV Rn, #data/MOV R5, #data/20230403-0007",
            "/static/registerbasedtrace/MOV Rn, #data/MOV R6, #data/20230403-0008",
            "/static/registerbasedtrace/MOV Rn, #data/MOV R7, #data/20230403-0009",
            "/static/registerbasedtrace/MOV Rn, A/MOV R0, A/20230314-0003",
            "/static/registerbasedtrace/MOV Rn, A/MOV R1, A/20230314-0004",
            "/static/registerbasedtrace/MOV Rn, A/MOV R2, A/20230314-0005",
            "/static/registerbasedtrace/MOV Rn, A/MOV R3, A/20230314-0006",
            "/static/registerbasedtrace/MOV Rn, A/MOV R4, A/20230314-0007",
            "/static/registerbasedtrace/MOV Rn, A/MOV R5, A/20230314-0008",
            "/static/registerbasedtrace/MOV Rn, A/MOV R6, A/20230314-0009",
            "/static/registerbasedtrace/MOV Rn, A/MOV R7, A/20230314-0010",
            "/static/registerbasedtrace/MOV Rn, direct/MOV R0, direct/20230403-0010",
            "/static/registerbasedtrace/MOV Rn, direct/MOV R1, direct/20230403-0011",
            "/static/registerbasedtrace/MOV Rn, direct/MOV R2, direct/20230403-0012",
            "/static/registerbasedtrace/MOV Rn, direct/MOV R3, direct/20230403-0013",
            "/static/registerbasedtrace/MOV Rn, direct/MOV R4, direct/20230403-0014",
            "/static/registerbasedtrace/MOV Rn, direct/MOV R5, direct/20230403-0015",
            "/static/registerbasedtrace/MOV Rn, direct/MOV R6, direct/20230403-0016",
            "/static/registerbasedtrace/MOV Rn, direct/MOV R7, direct/20230403-0017",
            "/static/registerbasedtrace/ORL A, Rn/ORL A, R0/20230314-0011",
            "/static/registerbasedtrace/ORL A, Rn/ORL A, R1/20230314-0012",
            "/static/registerbasedtrace/ORL A, Rn/ORL A, R2/20230314-0013",
            "/static/registerbasedtrace/ORL A, Rn/ORL A, R3/20230314-0014",
            "/static/registerbasedtrace/ORL A, Rn/ORL A, R4/20230327-0001",
            "/static/registerbasedtrace/ORL A, Rn/ORL A, R5/20230327-0002",
            "/static/registerbasedtrace/ORL A, Rn/ORL A, R6/20230327-0003",
            "/static/registerbasedtrace/ORL A, Rn/ORL A, R7/20230327-0004",
            "/static/registerbasedtrace/SUBB A, Rn/SUBB A, R0/20230327-0001",
            "/static/registerbasedtrace/SUBB A, Rn/SUBB A, R1/20230327-0001",
            "/static/registerbasedtrace/SUBB A, Rn/SUBB A, R2/20230327-0002",
            "/static/registerbasedtrace/SUBB A, Rn/SUBB A, R3/20230327-0003",
            "/static/registerbasedtrace/SUBB A, Rn/SUBB A, R4/20230327-0004",
            "/static/registerbasedtrace/SUBB A, Rn/SUBB A, R5/20230327-0005",
            "/static/registerbasedtrace/SUBB A, Rn/SUBB A, R6/20230327-0006",
            "/static/registerbasedtrace/SUBB A, Rn/SUBB A, R7/20230327-0007",
            "/static/registerbasedtrace/XCH A, Rn/XCH A, R0/20230327-0008",
            "/static/registerbasedtrace/XCH A, Rn/XCH A, R1/20230327-0009",
            "/static/registerbasedtrace/XCH A, Rn/XCH A, R2/20230327-0010",
            "/static/registerbasedtrace/XCH A, Rn/XCH A, R3/20230327-0011",
            "/static/registerbasedtrace/XCH A, Rn/XCH A, R4/20230327-0012",
            "/static/registerbasedtrace/XCH A, Rn/XCH A, R5/20230327-0013",
            "/static/registerbasedtrace/XCH A, Rn/XCH A, R6/20230327-0014",
            "/static/registerbasedtrace/XCH A, Rn/XCH A, R7/20230330-0001",
            "/static/registerbasedtrace/XRL A, Rn/XRL A, R0/20230327-0002",
            "/static/registerbasedtrace/XRL A, Rn/XRL A, R1/20230327-0003",
            "/static/registerbasedtrace/XRL A, Rn/XRL A, R2/20230327-0004",
            "/static/registerbasedtrace/XRL A, Rn/XRL A, R3/20230327-0005",
            "/static/registerbasedtrace/XRL A, Rn/XRL A, R4/20230327-0006",
            "/static/registerbasedtrace/XRL A, Rn/XRL A, R5/20230328-0001",
            "/static/registerbasedtrace/XRL A, Rn/XRL A, R6/20230328-0002",
            "/static/registerbasedtrace/XRL A, Rn/XRL A, R7/20230328-0003",
            ]

for i in range(len(FILENAME)):
    FILENAME[i] = Absolute_address + FILENAME[i]

# 寄存器名称
INSTRUCTION_REGNAME = ["ADD A, R1", "ADD A, R2", "ADD A, R3", "ADD A, R4", "ADD A, R5", "ADD A, R6", "ADD A, R7",
                       "ADDC A, R0", "ADDC A, R1", "ADDC A, R2", "ADDC A, R3", "ADDC A, R4", "ADDC A, R5", "ADDC A, R6",
                       "ADDC A, R7", "ANL A, R0", "ANL A, R1", "ANL A, R2", "ANL A, R3", "ANL A, R4", "ANL A, R5",
                       "ANL A, R6", "ANL A, R7", "DEC R0", "DEC R1", "DEC R2", "DEC R3", "DEC R4", "DEC R5", "DEC R6",
                       "DEC R7", "DJNZ R0,rel", "DJNZ R1,rel", "DJNZ R2,rel", "DJNZ R3,rel", "DJNZ R4,rel",
                       "DJNZ R5,rel", "DJNZ R6,rel", "DJNZ R7,rel", "INC R0", "INC R1", "INC R2", "INC R3", "INC R4",
                       "INC R5", "INC R6", "INC R7", "MOV A, R0", "MOV A, R1", "MOV A, R2", "MOV A, R3", "MOV A, R4",
                       "MOV A, R5", "MOV A, R6", "MOV A, R7", "MOV R0, #data", "MOV R1, #data", "MOV R2, #data",
                       "MOV R3, #data", "MOV R4, #data", "MOV R5, #data", "MOV R6, #data", "MOV R7, #data", "MOV R0, A",
                       "MOV R1, A", "MOV R2, A", "MOV R3, A", "MOV R4, A", "MOV R5, A", "MOV R6, A", "MOV R7, A",
                       "MOV R0, direct", "MOV R1, direct", "MOV R2, direct", "MOV R3, direct", "MOV R4, direct",
                       "MOV R5, direct", "MOV R6, direct", "MOV R7, direct", "ORL A, R0", "ORL A, R1", "ORL A, R2",
                       "ORL A, R3", "ORL A, R4", "ORL A, R5", "ORL A, R6", "ORL A, R7", "SUBB A, R0", "SUBB A, R1",
                       "SUBB A, R2", "SUBB A, R3", "SUBB A, R4", "SUBB A, R5", "SUBB A, R6", "SUBB A, R7", "XCH A, R0",
                       "XCH A, R1", "XCH A, R2", "XCH A, R3", "XCH A, R4", "XCH A, R5", "XCH A, R6", "XCH A, R7",
                       "XRL A, R0", "XRL A, R1", "XRL A, R2", "XRL A, R3", "XRL A, R4", "XRL A, R5", "XRL A, R6",
                       "XRL A, R7"]
# 寄存器数字标签
INSTRUCTION_REGLABEL = [11, 12, 13, 14, 15, 16, 17,
                        20, 21, 22, 23, 24, 25, 26, 27,
                        30, 31, 32, 33, 34, 35, 36, 37,
                        40, 41, 42, 43, 44, 45, 46, 47,
                        50, 51, 52, 53, 54, 55, 56, 57,
                        60, 61, 62, 63, 64, 65, 66, 67,
                        70, 71, 72, 73, 74, 75, 76, 77,
                        80, 81, 82, 83, 84, 85, 86, 87,
                        90, 91, 92, 93, 94, 95, 96, 97,
                        100, 101, 102, 103, 104, 105, 106, 107,
                        110, 111, 112, 113, 114, 115, 116, 117,
                        120, 121, 122, 123, 124, 125, 126, 127,
                        130, 131, 132, 133, 134, 135, 136, 137,
                        140, 141, 142, 143, 144, 145, 146, 147,
                        ]

# 指令字典：名称和对应数字标签
INSTRUCTION_DICT = {
    0: 'ADD A, Rn', 1: 'ADDC A, Rn', 2: 'ANL A, Rn', 3: 'DEC Rn', 4: 'DJNZ Rn,rel',
    5: 'INC Rn', 6: 'MOV A, Rn', 7: 'MOV Rn, #data', 8: 'MOV Rn, A', 9: 'MOV Rn, direct',
    10: 'ORL A, Rn', 11: 'SUBB A, Rn', 12: 'XCH A, Rn', 13: 'XRL A, Rn'
}
# 提取键和值到两个列表
INSTRUCTION_LABEL = list(INSTRUCTION_DICT.keys())
INSTRUCTION_NAME = list(INSTRUCTION_DICT.values())

# 每条指令采集的样本数
SAMPLES_NUM = 100
