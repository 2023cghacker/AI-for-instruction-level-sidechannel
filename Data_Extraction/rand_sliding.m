% 随机生成偏移量[-50，50]得到滑动窗口
def rand_sliding(H_power, instruction_no, offset, n):
% 构造一个滑动窗口
o = random.randint(-50, 50)  % 随机一个偏移量
% print("本次偏移量:", o)
new_t = H_level_t + o

% x_begin = H_level_t[0]
% x_end = H_level_t[len(H_level_t) - 1]
% x_begin = x_begin + o
% x_end = x_end + o

%     H_power(i,:)存储了第i次采样中指令触发时对应的功耗;
%     H_power是一个n * m矩阵，n是采样总数，m是指令触发时的时间段长度;
%     instruction_no是一个长为n的数组, 存储着该次采样的指令类型（0~20编号）
p = A[new_t].transpose().tolist()  % 转置并改成list类型
if n == 0:  % 第一次赋值
    H_power = p
    instruction_no = k
    offset = o
else:
    H_power = np.append(H_power, p, axis=0)  % 拼接
    instruction_no = np.append(instruction_no, k)
    offset = np.append(offset, o)
    n = n + 1
    
    return (H_power, instruction_no, offset, n)