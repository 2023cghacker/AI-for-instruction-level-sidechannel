"""
    @Author : ling chen
    @Create : 2024/12
    @Last modify: 2024/12
    @Description: 配置文件，定义了一些关于 trace_crossresistance 数据集的全局常量，包括数据文件地址的存储等
"""

import os

# 获取当前脚本文件的绝对路径
Absolute_address = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")

# 相对路径
DIRECTORY_PREFIX = Absolute_address + "/DataFile/DataSets_crossdevice/"

# 文件前缀名
FILENAME = ["/static/trace_crossresistance/llll/ADD/20240125-0001",
            "/static/trace_crossresistance/llll/ADDC/20240125-0002",
            "/static/trace_crossresistance/llll/ANL/20240125-0003",
            "/static/trace_crossresistance/llll/CLR/20240125-0004",
            "/static/trace_crossresistance/llll/CPL/20240125-0005",
            "/static/trace_crossresistance/rlll/ADD/20240125-0001",
            "/static/trace_crossresistance/rlll/ADDC/20240125-0002",
            "/static/trace_crossresistance/rlll/ANL/20240125-0003",
            "/static/trace_crossresistance/rlll/CLR/20240125-0004",
            "/static/trace_crossresistance/rlll/CPL/20240125-0005",
            "/static/trace_crossresistance/rrll/ADD/20240125-0001",
            "/static/trace_crossresistance/rrll/ADDC/20240125-0002",
            "/static/trace_crossresistance/rrll/ANL/20240125-0003",
            "/static/trace_crossresistance/rrll/CLR/20240125-0004",
            "/static/trace_crossresistance/rrll/CPL/20240125-0005",
            "/static/trace_crossresistance/rrrl/ADD/20240125-0001",
            "/static/trace_crossresistance/rrrl/ADDC/20240125-0002",
            "/static/trace_crossresistance/rrrl/ANL/20240125-0003",
            "/static/trace_crossresistance/rrrl/CLR/20240125-0004",
            "/static/trace_crossresistance/rrrl/CPL/20240125-0005",
            "/static/trace_crossresistance/rrrr/ADD/20240125-0001",
            "/static/trace_crossresistance/rrrr/ADDC/20240125-0002",
            "/static/trace_crossresistance/rrrr/ANL/20240125-0003",
            "/static/trace_crossresistance/rrrr/CLR/20240125-0004",
            "/static/trace_crossresistance/rrrr/CPL/20240125-0005", ]
for i in range(len(FILENAME)):
    FILENAME[i] = Absolute_address + FILENAME[i]

# 指令字典：名称和对应数字标签
INSTRUCTION_DICT = {
    0: 'ADD', 1: 'ADDC', 2: 'ANL', 3: 'CLR', 4: 'CPL'
}
# 提取键和值到两个列表
INSTRUCTION_LABEL = list(INSTRUCTION_DICT.keys()) * 5  # 5组设备电阻
INSTRUCTION_NAME = list(INSTRUCTION_DICT.values()) * 5

# 每条指令采集的样本数
SAMPLES_NUM = 200
