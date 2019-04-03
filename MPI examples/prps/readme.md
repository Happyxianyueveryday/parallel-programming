# 基于MPI的并行正则采样排序算法

## 0. 源代码
 源代码请参见本文件夹下的prps.c和prps2.c，其中prps2.c在prps.c基础上进行了一定的代码优化与改进。

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
 
 ```
 /* 1.设置进程数量，创建进程和获得当前进程编号*/
    int comm_sz=3;        /*进程数量*/
    int my_rank;           /*进程编号*/
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    
    /* 2.计算各个进程应该处理的数组元素数量，使用块划分法*/
    double *data=NULL;
    double *lis=NULL;
    int local_sz;      /*当前进程所处理的数组元素数量*/
    long int total_sz;      /*所有数组元素数量*/
    
    /*首先确定二进制文件中数字的个数*/
    FILE *fptr;
    fptr=fopen("psrs_data.dat", "r");
    fread(&total_sz,sizeof(long int),1,fptr);         /*读取文件中数据的总数为total_sz*/

    /*计算各个进程应当处理的数组元素个数，使用块划分法*/  
    local_sz=total_sz/comm_sz;

    /* 3.进程0进行读取数据，并将数据进行分段，将分段数据发送给其他进程，非0进程则从进程0接收其负责处理的数组元素部分*/
    if(my_rank==0)       
    {
        /*仅进程0申请所有数组元素所需要的空间，并从txt文件中读取所有数组元素*/
        data=(double *)malloc(total_sz*sizeof(double));   
        fread(data,sizeof(double),total_sz,fptr);
        /*进程0创建自身需要处理的数组*/
        lis=(double *)malloc(local_sz*sizeof(double));
        /*然后将数组按照上述划分的各个进程处理的元素的数量，将数组的各个片段散射到各个进程中*/
        MPI_Scatter(data,local_sz,MPI_DOUBLE,lis,local_sz,MPI_DOUBLE,0,MPI_COMM_WORLD);
        /*进程0的散射通信结束后，释放全部元素的数组data的空间*/
        free(data);
    }
    else
    {
        /*非0进程创建自身需要处理的数组*/
        lis=(double *)malloc(local_sz*sizeof(double));
        /*然后和进程0进行散射通信*/
        MPI_Scatter(data,local_sz,MPI_DOUBLE,lis,local_sz,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    
    /*并行正则采样排序算法的正式实现部分*/   /*本部分必须严格注意所创建的数组的大小*/
    /* 4.第一步：本地数据排序*/
    /*各个进程(包括进程0)对其负责的数组元素进行本地快速排序*/
    quickSort(lis,local_sz);      /*先对当前进程负责的数组元素进行排序*/
    double *samples=(double *)malloc(comm_sz*sizeof(double));      /*当前进程采样数组，当前进程采样数组的元素个数为进程数量comm_sz*/
    int i=0;
    for(i=0;i<comm_sz;i++)
    {
        samples[i]=lis[i*(local_sz/comm_sz)];     /*获得各个进程的进程采样数组，由于采样个数为comm_sz，因此采样步长为local_sz/comm_sz*/
    }

 ```
 



 
 

