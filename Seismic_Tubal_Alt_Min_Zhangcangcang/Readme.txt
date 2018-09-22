error文件夹中的两个m文件：
1.Tubal_Alt_Min.m求取不同面缺失情况下的恢复精度RSE.
2.Tubal_TNN.m比较Tubal_Alt_Min算法和TNN算法之间的恢复精度和收敛速度。

T_synthetic_tubal_rank_2.mat是合成的地震数据集，大小为：64*64*256,三个维度分别代表,inline,crossline,time。实际做面缺失的时候需要转换维度。

Tubal_Alt_Min.m文件是计算面缺失下的恢复精度。
