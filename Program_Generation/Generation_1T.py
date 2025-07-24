import random
import numpy as np

top_instruction = ["ORG 00H", "Start:", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "MOV P1,#0x04"]

bottom_instruction = ["MOV P1,#0x00", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "NOP", "JMP Start", "end"]

INSTRUCTION_NAME = ["ADD A, Rn", "ADDC A, Rn", "ANL A, Rn", "DEC Rn", "INC Rn", "MOV A, Rn",
                    "MOV Rn, #0x01", "MOV Rn, A", "ORL A, Rn", "SUBB A, Rn", "XCH A, Rn", "XRL A, Rn"]

TOTAL_INSTRUCTION = ["ADD A, R1", "ADD A, R2", "ADD A, R3", "ADD A, R4", "ADD A, R5", "ADD A, R6", "ADD A, R7",
                     "ADDC A, R0", "ADDC A, R1", "ADDC A, R2", "ADDC A, R3", "ADDC A, R4", "ADDC A, R5", "ADDC A, R6",
                     "ADDC A, R7", "ANL A, R0", "ANL A, R1", "ANL A, R2", "ANL A, R3", "ANL A, R4", "ANL A, R5",
                     "ANL A, R6", "ANL A, R7", "DEC R0", "DEC R1", "DEC R2", "DEC R3", "DEC R4", "DEC R5", "DEC R6",
                     "DEC R7", "INC R0", "INC R1", "INC R2", "INC R3", "INC R4",
                     "INC R5", "INC R6", "INC R7", "MOV A, R0", "MOV A, R1", "MOV A, R2", "MOV A, R3", "MOV A, R4",
                     "MOV A, R5", "MOV A, R6", "MOV A, R7", "MOV R0, #0x01", "MOV R1, #0x01", "MOV R2, #0x01",
                     "MOV R3, #0x01", "MOV R4, #0x01", "MOV R5, #0x01", "MOV R6, #0x01", "MOV R7, #0x01", "MOV R0, A",
                     "MOV R1, A", "MOV R2, A", "MOV R3, A", "MOV R4, A", "MOV R5, A", "MOV R6, A", "MOV R7, A",
                     "ORL A, R0", "ORL A, R1", "ORL A, R2",
                     "ORL A, R3", "ORL A, R4", "ORL A, R5", "ORL A, R6", "ORL A, R7", "SUBB A, R0", "SUBB A, R1",
                     "SUBB A, R2", "SUBB A, R3", "SUBB A, R4", "SUBB A, R5", "SUBB A, R6", "SUBB A, R7", "XCH A, R0",
                     "XCH A, R1", "XCH A, R2", "XCH A, R3", "XCH A, R4", "XCH A, R5", "XCH A, R6", "XCH A, R7",
                     "XRL A, R0", "XRL A, R1", "XRL A, R2", "XRL A, R3", "XRL A, R4", "XRL A, R5", "XRL A, R6",
                     "XRL A, R7"]

INSTRUCTION_LEN = np.ones(95)
for i in range(len(INSTRUCTION_LEN)):
    print(i, " ", TOTAL_INSTRUCTION[i], " ", INSTRUCTION_LEN[i])

"""
生成程序，并存入txt文件
"""

with open("output.txt", "w") as file:
    '''
    1.将程序开头写入
    '''
    for line in top_instruction:
        file.write(line + "\n")

    '''
    2.生成程序中部的随机片段(200次)，然后进行写入，每个片段如下:
    rand ins (随机指令)
    target ins (目标指令)
    rand ins (随机指令)
    '''
    target_ins = "ADD A,R0"
    target_index = np.zeros(200)
    index = 0
    for i in range(200):
        # 前一个随机指令
        rand_num1 = random.randint(0, 94)
        rand_ins1 = TOTAL_INSTRUCTION[rand_num1]
        index = index + INSTRUCTION_LEN[rand_num1]  # 索引往后位移
        file.write(rand_ins1 + "\n")

        # 中间的目标指令
        target_index[i] = index
        # print(index, end=" ")
        index = index + 1  # 索引往后位移
        file.write(target_ins + "\n")

        # 后一个随机指令
        rand_num2 = random.randint(0, 94)
        rand_ins2 = TOTAL_INSTRUCTION[rand_num2]
        index = index + INSTRUCTION_LEN[rand_num2]  # 索引往后位移
        file.write(rand_ins2 + "\n")

    '''
    3.将程序结尾写入
    '''
    for line in bottom_instruction:
        file.write(line + "\n")

# print(target_index)
