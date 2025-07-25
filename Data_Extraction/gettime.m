%%是findtime的封装形式，作为一个函数被data_preprocessing调用%%
function  [H_level_t,begin_t]=gettime(k)

filename=["../static/DATA_100per/ADD/20220409-0003", "../static/DATA_100per/ADDC/20220409-0021",...
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

% disp(filename(k))
load(filename(k)); %%导入第k个指令的首个mat文件

H_level_t=find(B>0.2==1); %找出高电平对应的时间段
% L_level_t=find(B<0.2==1); %找出低电平对应的时间段
begin_t = H_level_t(1);  % 高电平的起始时间，即指令片段的起始时间
