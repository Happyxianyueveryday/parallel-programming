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

### A. 时间T分析
 时间T的测定结果如下表所示。
![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/prps/pics/QQ%E6%88%AA%E5%9B%BE20190403210316.png)

  + 对时间T的分析：测定的运行时间在某些方面是出乎我的预期的。由上表可见，随着待排序数据量的增大，排序时间稳定增加，这是符合预期的。但是随着进程数量的增加，排序时间先减少后增加，而且在进程数为64和128时的排序时长明显增加。个人推测的一个原因是因为个人代码实现原因，进程之间通信的代价已经抵消了进程数增加对性能的提升。
 
### B. 加速比S分析
 加速比S的定义公式如下。  

![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/prps/pics/QQ%E6%88%AA%E5%9B%BE20190403211408.png)

 因为所编写的PRPS程序并不能在进程为1时合法运行（因为使用了进程通信），因此我们这里只是将进程为2的情况视作串行情况，加速比S的测定结果如下表所示。
![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/prps/pics/QQ%E6%88%AA%E5%9B%BE20190403210330.png)

### C. 效率T分析
 效率T的定义公式如下。  

![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/prps/pics/QQ%E6%88%AA%E5%9B%BE20190403211417.png)

 同加速比的情况，我们这里将进程为2的情况视作串行情况，效率T的测定结果如下表所示。
![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/prps/pics/QQ%E6%88%AA%E5%9B%BE20190403210342.png)


## 3. 具体实现过程
### 步骤1：本地数据排序：
 首先，在进行并行正则归并排序之前，首先需要读取文件中的数据，由给定的格式可知二进制数据文件中的第一个八字节long int为待排序数据的个数。后续的字节则为待排序数组的元素。因此首先由进程0读取文件中的首8个字节，将读取到的数字设定为数组的大小，然后进程0以8个字节为一组依次读取剩下的数组元素。
 然后，在读取完成数据后，假设进程数量为comm_sz，首先按照块划分法将数组分为comm_sz份，进程0负责进行划分，然后使用MPI集合通信——MPI_Bcast将各个部分广播给各个进程。每个进程接收到数据后，首先对自己所需要处理的数组部分进行快速排序，并得到本地数据排序结果。
 最后，各个进程对本地数据进行采样，具体而言，将本地数据分为comm_sz份，每份的第一个数组元素作为样本。在进行上述采样后得到一个元素数量为comm_sz个的样本数据。
 以数据数量total_sz=18且进程数量comm_sz=3为例，步骤1——本地排序过程的具体过程如下图所示。



 
 

