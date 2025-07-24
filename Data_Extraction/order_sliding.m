%    按序从[-50，50]生成偏移量得到滑动窗口

function  [H_power, instruction_no, offset, n]= order_sliding(H_power, instruction_no, offset, n,A,H_level_t,k)
for o = -50: 49 %按序生成偏移量
    % print("本次偏移量:", o)
    new_t = H_level_t + o;
    %         H_power(i,:)存储了第i次采样中指令触发时对应的功耗;
    %         H_power是一个n * m矩阵，n是采样总数，m是指令触发时的时间段长度;
    %         instruction_no是一个长为n的数组, 存储着该次采样的指令类型（0~20编号）
    p = A(new_t);
    
    H_power=[H_power;p];
    instruction_no = [instruction_no;k];
    offset = [offset;o];
    %     if n == 0:  % 第一次赋值
    %         H_power = p
    %         instruction_no = k
    %         offset = o
    %     else
    %         H_power = np.append(H_power, p, axis=0)  % 拼接
    %         instruction_no = np.append(instruction_no, k)
    %         offset = np.append(offset, o)
    %     end
    n = n + 1;
end
% return (H_power, instruction_no, offset, n)
