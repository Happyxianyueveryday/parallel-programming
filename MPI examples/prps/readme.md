# 基于MPI的并行正则采样排序算法

## 0. 源代码
 源代码请参见本文件夹下的prps.c和prps2.c，感谢某位dalao的反馈，对代码进行了重构和优化，改进了编写风格，其中prps2.c在prps.c基础上进行了一定的代码优化与改进。

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
 
 
 ### 步骤2：获得划分主元
 首先，由进程0收集各个进程中的采样数组，因为每个进程中的采样数组的长度均为comm_sz，因此这一过程可以直接使用MPI集合通信——MPI_Gather来完成。
	然后，进程0将收集的来自各个进程的采样数组按顺序合并，合并后的采样数组长度为comm_sz\*comm_sz，并进行快速排序。接着按一组comm_sz个元素将采样数组分为comm_sz组，除掉第一组外，取每一组的第一个数组元素组成新的采样数组，该新采样数组的长度为comm_sz-1。
	最后，进程0将新采样数组分别分发给所有进程，由于新采样数组的长度固定为comm_sz-1，因此可以直接使用MPI集合通信——MPI_Bcast来完成。
	
 ```
 /* 5.第二步：获得划分主元*/
    /*首先，由进程0收集所有主元，并对所有主元进行排序，然后按进程数N对全体样本等间隔采样，最后将采样主元散射给其他进程*/
    double *pivot=NULL;            /*所有进程的主元数组，每个进程的采样主元数组的大小为comm_sz，因此该主元数组的元素个数为comm_sz*comm_sz*/
    double *sample_pivot=NULL;     /*采样主元数组，该采样主元数组的元素个数为comm_sz-1*/
    if(my_rank==0)
    {
        /*首先收集所有主元到数组pivot中*/    
        pivot=(double *)malloc((comm_sz*comm_sz)*sizeof(double));        /*申请主元数组空间*/
        MPI_Gather(samples,comm_sz,MPI_DOUBLE,pivot,comm_sz,MPI_DOUBLE,0,MPI_COMM_WORLD);    /*进行收集操作*/
        /*然后对数组pivot进行排序*/
        quickSort(pivot,comm_sz*comm_sz);
        /*之后对主元数组进行按进程数comm_sz的等间隔采样，并除去下标为0的首元素，从而获得comm_sz-1个采样主元*/
        sample_pivot=(double *)malloc((comm_sz-1)*sizeof(double));    /*申请采样主元数组空间*/
        int i;
        for(i=0;i<comm_sz-1;i++)
        {
            sample_pivot[i]=pivot[(i+1)*comm_sz];
        }
        /*最后将采样主元数组直接广播给其他进程*/
        MPI_Bcast(sample_pivot,comm_sz-1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    else    /*其他进程只需要和进程0进行集合通信操作*/
    {
        MPI_Gather(samples,comm_sz,MPI_DOUBLE,pivot,comm_sz,MPI_DOUBLE,0,MPI_COMM_WORLD);   /*配合进程0的聚集操作，向进程0发送当前进程主元数组samples*/
        sample_pivot=(double *)malloc((comm_sz-1)*sizeof(double));    /*申请采样主元数组空间*/
        MPI_Bcast(sample_pivot,comm_sz-1,MPI_DOUBLE,0,MPI_COMM_WORLD); /*配合进程0的广播操作，从进程0接收采样主元数组sample_pivot*/
    }

 ```
 
### 步骤3：交换数据
 回顾一下步骤2，在步骤2结束后，每个进程都得到一个含有comm_sz-1个元素的新采样数组。
	首先，每个进程将新采样数组中的comm_sz-1个元素插入排序后的本地数据中，使得每个插入元素左侧的元素均小于该插入元素，每个插入数组右侧的元素均大于该插入元素。插入完成后，插入的comm_sz-1个元素将每个进程的本地数据划分为comm_sz组。
	然后，依次将本地数据被划分出的comm_sz组分别编号0,1,2…,comm_sz-1，然后依次将每个进程中的第i组分发给进程i。这一过程使用MPI集合通信——MPI_Alltoallv来完成。需要特别注意的是，在使用集合通信MPI_Alltoallv之前，首先需要统计各个进程中的各个分段的长度。
 
 ```
 /* 6.第三步：交换数据*/
    /*各个进程(包括进程0)使用原有的或者由广播获得的样本主元数组sample_pivot的comm_sz-1个元素将本地数组进行划分，划分为comm_sz段，特别注意到这时本地数组lis已经经过排序；然后进行全互换操作*/
    /*我们使用两个数组来记录各个进程中的原始数组的各个分段数组的信息，一是数组depart，该数组记录了comm_sz个数组分段各自的起始下标；二是数组depart_sz，该数组记录了各个数组分段各自的元素个数*/
    int *depart=(int *)malloc(comm_sz*sizeof(int));    /*数组的各个分段的首元素下标*/
    int *depart_sz=(int *)malloc(comm_sz*sizeof(int)); /*数组的各个分段的元素个数*/
    /*计算分段首元素下标*/
    depart[0]=0;      /*第一个分段的首元素下标为0*/
    int k;
    for(i=1;i<comm_sz;i++)
    {
        for(k=0;k<local_sz;k++)
        {
            if(lis[k]>sample_pivot[i-1])
            {
                depart[i]=k;
                break;
            }
        }
    }
    /*计算分段元素个数*/
    for(i=0;i<comm_sz;i++)
    {
        if(i==comm_sz-1)
        depart_sz[i]=local_sz-depart[i];
        else
        depart_sz[i]=depart[i+1]-depart[i];
    }
    /*计算出分隔点的位置后，进行全互换操作，该步骤的难度较大，且是本次实验的核心内容*/
    /*第一步：各个进程需要将自己的各个分段的长度，通知所有的其他进程，并且从所有的其他进程收集其他段的长度，这个过程使用Allgather集合通信来进行实现*/
    int *size_dic=(int *)malloc((comm_sz*comm_sz)*sizeof(int));   /*各个进程的分段长度记录*/
    MPI_Allgather(depart_sz,comm_sz,MPI_INT,size_dic,comm_sz,MPI_INT,MPI_COMM_WORLD);    
    /*第二步：各个进程将当前进程中的各个分段生成二维数组lis_slot*/
    double **lis_slot=(double **)malloc(comm_sz*sizeof(double *));      
    for(i=0;i<comm_sz;i++)
    {
        lis_slot[i]=(double *)malloc(local_sz*sizeof(double));      /*每个分段的长度最多为local_sz，单个进程中的分段的数量为comm_sz*/
    }
    int p;
    for(i=0;i<comm_sz;i++)
    {
        for(k=depart[i],p=0;k<depart[i]+depart_sz[i];k++,p++)
        {
            lis_slot[i][p]=lis[k];
        }
    }
    /*第三步：开始进行收集，即进程i从其他所有进程中收集分段i*/
    /*首先，创建分段收集数组final_lis，总共收集comm_sz个分段，每个分段最多local_sz个数据*/
    double *final_lis=(double *)malloc((comm_sz*local_sz)*sizeof(double));    
    /*然后进行收集操作，每个进程my_rank从所有进程(包括自身)中收集对应的第my_rank个分段*/
    /*分步1：制作每个进程接收的个数表和各个进程数据收集起始下标的偏移表*/
    int *sendcounts=(int *)malloc(comm_sz*sizeof(int));
    int *displs=(int *)malloc(comm_sz*sizeof(int));
    /*分步2：扁平化lis_slot*/
    double *lis_slot_flatten=(double *)malloc(comm_sz*local_sz*sizeof(double));
    int m=0;
    for(i=0;i<comm_sz;i++)
    {
        for(k=0;k<local_sz;k++)
        {
            lis_slot_flatten[m]=lis_slot[i][k];
            m++;
        }
    }
    for(i=0;i<comm_sz;i++)
    {
        sendcounts[i]=local_sz;
        displs[i]=local_sz*i;
    }
    MPI_Alltoallv(lis_slot_flatten,sendcounts,displs,MPI_DOUBLE,final_lis,sendcounts,displs,MPI_DOUBLE,MPI_COMM_WORLD);
 ```
 
### 步骤4：归并排序
 在步骤3完成后，每个进程i都得到了所有进程中的本地数据的编号为i的分块。首先，将这comm_sz个分块进行经典的comm_sz路归并，从而合并为一个归并结果数组。
	然后，将各个进程中的归并结果数组按照进程编号从小到大的顺序合并为一个数组，该数组即为最终结果。因为各个进程中的归并结果数组长度不同，我们考虑使用MPI集合通信——MPI_Gatherv将各个进程中的归并结果收集到进程0中，最终得到合并结果。
 
 ```
 /* 7.归并排序*/
    /*首先这里简单做一下回顾，一维数组final_lis中存储了所有的comm_sz个分段，只需要将这comm_sz个分段进行归并(直接归并，无需排序)即可，每个分段中总共有local_sz个数据*/
    /*首先，由size_dic生成当前进程my_rank收集的各个分段数组的元素数量，生成结果为size_my_dic*/
    int *size_my_dic=(int *)malloc(comm_sz*sizeof(int));   /*当前进程my_rank接收的各个分段数组的元素数量*/
    for(i=0;i<comm_sz;i++)
    {
        size_my_dic[i]=size_dic[i*comm_sz+my_rank];
    }
    int result_lis_sz=0;          /*归并排序结果数组的大小*/
    for(i=0;i<comm_sz;i++)
    {
        result_lis_sz+=size_my_dic[i];
    }
    double *result_lis=(double *)malloc(result_lis_sz*sizeof(double));   /*创建归并数组排序结果数组result_lis*/
    mergeSort(result_lis,result_lis_sz,final_lis,local_sz,size_my_dic,my_rank,comm_sz);        /*对收到的各个分段进行归并排序*/
    
    
    /*8.汇总结果*/
    /*将经过排序的各个进程中的结果使用集合通信Gatherv将result_lis汇总到进程0，由进程0输出结果*/
    /*首先使用size_dic计算各个进程中的归并排序结果中的元素个数，以及在最终汇总结果中的偏移量*/
    int *merge_size=(int *)malloc(comm_sz*sizeof(int));
    int *merge_disp=(int *)malloc(comm_sz*sizeof(int));
    memset(merge_size,0,comm_sz*sizeof(int));
    memset(merge_disp,0,comm_sz*sizeof(int));
    for(i=0;i<comm_sz;i++)
    {
        for(k=0;k<comm_sz;k++)
        {
            merge_size[i]+=size_dic[k*comm_sz+i];
        }
    }
    for(i=1;i<comm_sz;i++)
    {
        merge_disp[i]=merge_disp[i-1]+merge_size[i-1];
    }
    /*然后直接使用集合通信Gatherv收集所有进程中的数据*/
    double *final_result=(double *)malloc(total_sz*sizeof(double));
    MPI_Gatherv(result_lis,merge_size[my_rank],MPI_DOUBLE,final_result,merge_size,merge_disp,MPI_DOUBLE,0,MPI_COMM_WORLD);
    
    /*最后，由进程0输出最终的排序结果*/
    if(my_rank==0)
    {
        for(i=0;i<total_sz;i++)
        {
            printf("%lf\n",final_result[i]);
        }
    }

 ```


 



 
 

