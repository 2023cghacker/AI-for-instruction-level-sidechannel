%%提取出指令（触发高电平）对应的功耗片段%%
clear;
clc;

n=1;
H_power=zeros(1,250); %初始化

for k=1:21 %k表示要对第几个指令（一共21个）进行提取。
    
    [H_level_t,L_level_t,t]=gettime(k); %调用函数gettime
    
    filename=["..\..\ADD\20220409-0003 (","..\..\ADDC\20220409-0021 (","..\..\ANL\20220409-0012 (",...
        "..\..\CLR\20220409-0008 (","..\..\CPL\20220409-0016 (","..\..\DA\20220409-0023 (",...
        "..\..\DEC\20220409-0005 (","..\..\DIV\20220409-0026 (","..\..\INC\20220409-0004 (",...
        "..\..\JMP\20220409-0007 (","..\..\MOV\20220409-0002 (","..\..\MUL\20220409-0025 (",...
        "..\..\NOP\20220409-0024 (","..\..\ORL\20220409-0013 (","..\..\POP\20220409-0010 (",...
        "..\..\PUSH\20220409-0009 (", "..\..\SETB\20220409-0015 (","..\..\SUBB\20220409-0022 (",...
        "..\..\SWAP\20220409-0006 (","..\..\XCH\20220409-0011 (", "..\..\XRL\20220409-0014 ("];
    
    for i=2:112 %对每一个指令统计其中111条采集的功率样本。
        fname=filename(k)+num2str(i)+").mat";
        load (fname);
        
        %H_power(i,:)存储了第i次采样中指令触发时对应的功耗;
        %H_power是一个n*m矩阵，n是采样总数，m是指令触发时的时间段长度;
        %instruction_num 是一个长为n的数组,存储着该次采样的指令类型（1~21编号）
        H_power(n,1:length(H_level_t))=A(H_level_t);
        instruction_num(n)=k; 
        n=n+1;
    end
    
end

X=H_power;               %输入
Y=instruction_num';      %输出

save('DATA','X','Y');

