/*coding utf-8*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <memory.h>
#include <limits.h>

/*方法compare：供qsort函数使用的比较函数*/
int compare(const void *a,const void *b)
{
    return *(double *)a>*(double *)b;
}

/*方法quickSort：该方法用于在各个进程中进行本地快速排序*/
/*参数列表：lis--当前进程进行本地数据排序的数组，大小为local_sz，local_sz--需要进行快速排序的数组的大小*/
void quickSort(double lis[],int local_sz)
{
    qsort(lis,local_sz,sizeof(double),compare);
}

/*方法judge：该方法判断归并排序中，是否所有元素均已经归并完毕*/
/*参数列表：size_my_dic--各路的元素个数，pos--各路游标指针，comm_sz归并路数*/
int judge(int *size_my_dic,int *pos,int comm_sz)
{
    int i;
    int flag=0;
    for(i=0;i<comm_sz;i++)
    {
        if(pos[i]<size_my_dic[i])
        flag=1;
    }
    return flag;
}

/*方法min：该方法判断归并排序中，各路游标所指向的元素中最小的一个*/
/*参数列表：size_my_dic--各路的元素个数，pos--各路游标指针，convert_lis--各路原始数据，comm_sz归并路数*/
int min(int *size_my_dic,int *pos,double **convert_lis,int comm_sz)
{
    int min_value=INT_MAX;   /*多路中的指针指向的最小值*/
    int min_pos=0;           /*最小值出现的路编号*/
    int i;
    for(i=0;i<comm_sz;i++)
    {
        if(pos[i]<size_my_dic[i]&&convert_lis[i][pos[i]]<min_value)
        min_value=convert_lis[i][pos[i]];
    }
    for(i=0;i<comm_sz;i++)
    {
        if(convert_lis[i][pos[i]]==min_value)
        min_pos=i;
    }
    pos[min_pos]++;
    return min_value;
}

/*方法mergeSort：该方法用于在各个进程收到来自其他进程的给定分段后，将各个分段进行归并排序*/
/*参数列表：iinal_lis--一维形式表示的二维数组，local_sz--final_lis中的每个分段的形式(存储)长度，size_my_dic--每个分段的实际长度，my_rank--当前进程的进程编号*/
/*这里实际上就是一个多路归并问题，这里先给出一个较为简单的循环遍历方法来进行多路归并*/
void mergeSort(double *result_lis,int result_lis_sz,double *final_lis,int local_sz,int *size_my_dic,int my_rank,int comm_sz)
{
    int *pos=(int *)malloc(comm_sz*sizeof(int));      /*各路的指针*/
    memset(pos,0,comm_sz*sizeof(int));      /*初始化各路指针为0*/
    /*首先，将一维数组final_lis根据段长local_sz转化为二维数组*/
    double **convert_lis=(double **)malloc(comm_sz*sizeof(double *));
    int i;
    for(i=0;i<comm_sz;i++)
    {
        convert_lis[i]=&final_lis[i*local_sz];   
    }   
    /*多路归并算法*/
    /*首先求解pos中comm_sz个指针指向中的较小值*/
    int p=0;
    while(judge(size_my_dic,pos,comm_sz))
    {
        int min_value=min(size_my_dic,pos,convert_lis,comm_sz);
        result_lis[p]=min_value;
        p++;
    }
    return;
}

/*mergeSort附属方法judge：该方法判断当前存在*/

/*主函数*/
int main(void)
{
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
        /*data[0]=48;data[1]=39;data[2]=6;data[3]=72,data[4]=91,data[5]=14;data[6]=69;data[7]=40;data[8]=89;data[9]=61;data[10]=12;data[11]=21;data[12]=84;data[13]=58;data[14]=32;data[15]=33;data[16]=72;data[17]=20;*/
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
	/*test
	if(my_rank==1)
	{
		for(i=0;i<comm_sz*local_sz;i++)
		{
			printf("%lf\n",lis_slot_flatten[i]);
		}
	}
	*/
	for(i=0;i<comm_sz;i++)
	{
		sendcounts[i]=local_sz;
		displs[i]=local_sz*i;
	}
	MPI_Alltoallv(lis_slot_flatten,sendcounts,displs,MPI_DOUBLE,final_lis,sendcounts,displs,MPI_DOUBLE,MPI_COMM_WORLD);
	
	
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
    
    MPI_Finalize();
    return 0;
}

