# 基于MPI的并行正则采样排序算法

## 1. 并行正则采样排序算法简介

并行正则采样排序是最经典的并行排序算法，该算法是一种均匀划分的负载平衡的并行排序算法，相对于快速排序而言，其算法复杂度的上界和下界基本一致。

### 第一步：本地数据排序
 +  按进程数N等间隔采样
 +  对每一部分的采样片段进行排序
 
 ![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/prps/pics/%E5%9B%BE%E7%89%871.png)

### 第二步：获得划分主元 
 +  一个进程收集样本(用MPI集合通信)并对所有样本进行排序
 +  按进程数N对全体样本等间隔采样
 +  散发最终样本(用MPI集合通信)，即主元
 
 ![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/prps/pics/%E5%9B%BE%E7%89%872.png)
 
### 第三步：交换数据 
 +  交换本地数据划分后的分块
 +  进行全互换过程（用MPI集合通信） 
 
 ![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/prps/pics/%E5%9B%BE%E7%89%873.png)

### 第四步：归并排序
 +  合并得到的分块
 +  对最终的本地数据进行排序
 
 ![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/prps/pics/%E5%9B%BE%E7%89%874.png)
 
## 2. 性能分析
经过实验室提供的超算，在较小范围内进行测试的性能分析结果均用类似下面的表格表示。N=2^n为待排序的数组的元素个数，p为使用的CPU核心数。

 ![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/prps/pics/QQ%E6%88%AA%E5%9B%BE20190403210658.png)

### A. 加速比S分析
加速比S的测定结果如下表所示。
![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/prps/pics/QQ%E6%88%AA%E5%9B%BE20190403210316.png)


 


 
 

