# AI-for-instruction-level-sidechannel

## 介绍

侧信道指令级逆向
1.时钟周期（振荡周期/晶振周期/一个节拍）：通常为节拍脉冲或Ｔ周期，既主频的倒数，它是处理操作的最基本的单位。用P表示
2.状态周期：两个节拍定义为一个状态周期，用S表示
3.机器周期：为了便于管理，常把一条指令的执行过程划分为若干个阶段，每一阶段完成一项工作。
例如，取指令、存储器读、存储器写等，这每一项工作称为一个基本操作。完成一个基本操作所需要的时间称为机器周期。
4.指令周期：单片机从内存取出一条指令并执行这条指令的时间总和。一般由若干个机器周期组成
例如：AT89S52的一个机器周期等于12个时钟周期


## 安装教程
* 拉取下来的项目名称为AI-for-instruction-level-sidechannel，推荐改名src再运行，
因为原项目是放在src下面的，其余文件路径可以查看Configuration_0n.py文件

* 环境：略


## 项目架构
#### static
存放原始数据，原始数据就是从示波器采集下来的功耗，该数据以mat文件形式保存，里面有多个变量
Git上没有上传这个文件夹，因为太大了。
#### Data_Extraction
该文件夹中的代码负责提取原始数据，得到可供训练的功耗迹
#### DataFile
该文件夹存储各种数据文件，包括数据集，训练集测试集，神经网络等等
#### figure
该文件夹存储一些绘制的图片
#### gui
一些可视化代码
#### Model_MainCode
项目的核心代码，包含各类神经网络API，训练测试主程序等
#### Program_Generation
设计汇编程序
#### testfile
临时测试文件存放处，没有什么用
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip uninstall pillow
pip install pillow


## 参与贡献
#### 作者：凌琛

1. Fork 本仓库
2. 新建 master 分支
3. 提交代码
4. 新建 Pull Request

