# 基于MPI的并行正则采样排序算法

## 1. 并行正则采样排序算法简介

并行正则采样排序是最经典的并行排序算法，该算法是一种均匀划分的负载平衡的并行排序算法，相对于快速排序而言，其算法复杂度的上界和下界基本一致。

### 第一步：本地数据排序
 +  按进程数N等间隔采样
 +  对每一部分的采样片段进行排序
 
 ![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/prps/pics/%E5%9B%BE%E7%89%871.png)

### 第二步： 
 
 

