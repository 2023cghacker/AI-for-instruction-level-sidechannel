clear;
clc;

%准备工作: 读取和计算一些常量
FILENAME=["../static/DATA_100per/ADD/20220409-0003", "../static/DATA_100per/ADDC/20220409-0021",...
    "../static/DATA_100per/ANL/20220409-0012","../static/DATA_100per/CLR/20220409-0008", ...
    "../static/DATA_100per/CPL/20220409-0016","../static/DATA_100per/DA/20220409-0023",...
    "../static/DATA_100per/DEC/20220409-0005", "../static/DATA_100per/DIV/20220409-0026",...
    "../static/DATA_100per/INC/20220409-0004", "../static/DATA_100per/JMP/20220409-0007", ...
    "../static/DATA_100per/MOV/20220409-0002","../static/DATA_100per/MUL/20220409-0025",...
    "../static/DATA_100per/NOP/20220409-0024","../static/DATA_100per/ORL/20220409-0013",...
    "../static/DATA_100per/POP/20220409-0010","../static/DATA_100per/PUSH/20220409-0009",...
    "../static/DATA_100per/SETB/20220409-0015","../static/DATA_100per/SUBB/20220409-0022",...
    "../static/DATA_100per/SWAP/20220409-0006","../static/DATA_100per/XCH/20220409-0011",...
    "../static/DATA_100per/XRL/20220409-0014"];

INSTRUCTION_NAME = ["ADD", "ADDC", "ANL", "CLR", "CPL", "DA", "DEC", "DIV", "INC", "JMP", "MOV", "MUL", "NOP",...
    "ORL", "POP", "PUSH", "SETB", "SUBB", "SWAP", "XCH", "XRL"]
DIRECTORY_PREFIX = "../DataFile/SmallDataSets/"


file_pre = FILENAME  % 读取文件的地址前缀
instruction_name = INSTRUCTION_NAME   % 读取每个指令名
[H_level_t, begin_t] = gettime(1)   % 调用函数gettime，获取第一个指令的单位时间长度，并以此作为基准，以后都使用此长度进行提取指令
m = length(H_level_t)  % 指令的时间段长度，在小型数据集中m=125，在大型数据集中m=1500


%开始循环提取样本中的指令（功率片段），
%最后存入X,Y中用于机器学习
H_power = []
instruction_no = []
offset = []
n = 0
samples_num = 100  %SAMPLES_NUM  % 每条指令采集的样本数
for k =1:length(instruction_name)  % k表示要对第几个指令（一共21个）进行提取
    disp(['》》》正在分析提取第', num2str(k), '个指令中,指令名称:', num2str(instruction_name(k)), '《《《'])
    
    [H_level_t, begin_t] = gettime(k);   % 调用函数gettime,获取指令k的高电平时间段,起始时间点
    H_level_t = begin_t:begin_t + m - 1;  % 重新计算H_level_t为从起始时间begin_t开始，长度为m的时间段
    
    t = 1;
    fprintf("进度条[")
    for i =2:1 + samples_num  % 对该指令统计其中samples_num条采集的功率样本。
        if i > t * samples_num / 20
            fprintf('#'); % % 不换行
            t = t + 1;
        end
        
        fname = file_pre(k) + " (" + num2str(i) + ").mat";
        load(fname)  %% 读取Mat文件, 文件里有两个数组A,B
        
        %         % 随机生成滑动窗口
        %[ H_power, instruction_no, offset, n] = rand_sliding(H_power, instruction_no, offset, n,A,H_level_t,k)
        
        %         % 按序生成滑动窗口
        [H_power, instruction_no, offset, n] = order_sliding(H_power, instruction_no, offset, n,A,H_level_t,k);
    end
    fprintf("]\n\n")
end

% 这个文件最后得到的数据集为： 功率片段X与对应的指令Y
% 然后就可以开始跑机器学习了

X = H_power  % 输入,是一个n*m(m=125)的矩阵
Y = instruction_no  % 输出
print(offset)
% % 将输入X,输出Y,片段长度length都存入DATA.mat
add_pre = DIRECTORY_PREFIX  % 目录地址前缀
scipy.io.savemat(add_pre + 'rS_125d_DATA.mat', {'X': X, 'Y': Y, 'offset': offset})


