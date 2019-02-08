#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

int my_sqrt(int num)
{
    int i;
    for(i=1;i<=num;i++)
    {
        if(num%i==0&&num/i==i)
        return i;
    }
    return 0;
}

int main(void)
{
    /*推荐在每一次使用关键变量时，都附注上该变量的用途和含义，防止出现变量使用混淆，本实验过程中犯了不少变量使用混淆的问题*/
    int comm_sz;          /*进程总数*/
    int my_rank;          /*当前进程编号*/
    int i,k,p,q;          /*计数变量*/
    int local_sz;         /*矩阵分块的大小/元素个数*/
    int local_len;        /*矩阵分块的边长/行数/列数*/
    double *local_mat=NULL;    /*当前进程中接收到的矩阵块*/
    double *local_vec=NULL;    /*当前进程中接收到的向量块*/
    double *local_sum=NULL;    /*0号进程中的最终结果向量*/
    double *local_temp=NULL;
    double *mat=NULL;          /*输入矩阵*/
    double **initmat=NULL;     /*初始二维矩阵*/
    double *vec=NULL;          /*输入向量*/
    double *local_res;         /*单个进程的分块矩阵和分块向量相乘的结果*/
    int n;                /*矩阵和向量的维数*/
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    /*我们将本实验分为以下几个主要步骤来完成*/
    /*1. 进程0读取向量输入和矩阵输入，并将矩阵进行分块，并且将各个矩阵分块发送给各个进程*/
    if(my_rank==0)
    {
        /*首先，进程0读取输入的方阵和向量大小，方阵的行数，列数以及向量维数相等*/
        printf("Enter the number of matrix rows/cols:\n");
        scanf("%d",&n);
        initmat=(double **)malloc(n*sizeof(double *));    /*输入矩阵*/
        for(i=0;i<n;i++)
        {
            initmat[i]=(double *)malloc(n*sizeof(double));
        }
        vec=(double *)malloc(n*sizeof(double));           /*输入向量*/

        /*然后，进程0读取输入的矩阵和向量*/
        printf("Enter the matrix:\n");
        for(i=0;i<n;i++)
        {
            for(k=0;k<n;k++)
            {
                scanf("%lf",&initmat[i][k]);    /*注意，由于本实验中需要进行按列划分，因此二维数组的n行分别存储输入矩阵的n列*/
            }
        }
        printf("Enter the vector:\n");
        for(i=0;i<n;i++)
        {
            scanf("%lf",&vec[i]);     
        }

        /*接着进程0将二维矩阵转化为各进程的子矩阵的一维矩阵*/
        mat=(double *)malloc(n*n*sizeof(double));
        int local_len=n/my_sqrt(comm_sz);     
        int r=0;
        for(i=0;i<my_sqrt(comm_sz);i++)       
        {
            for(k=0;k<my_sqrt(comm_sz);k++)
            {
                /*生成分块矩阵的一维序列*/
                for(p=local_len*i;p<local_len*(i+1);p++)
                {
                    for(q=local_len*k;q<local_len*(k+1);q++)
                    {
                        mat[r]=initmat[p][q];   
                        r++;     
                    }
                }
            }
        }
        /*现在mat已经是子矩阵分块转化而成的一维数组，可以开始进行分发操作*/

        /*然后，进程0将矩阵分为comm_sz个子矩阵并且分别分发到各个进程中，每个进程得到一个子矩阵，并且将向量的对应分块发送到各个进程中*/
        MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);                  /*进程0首先向其他进程广播矩阵和向量的维数*/
        local_sz=n*n/comm_sz;                                      /*每一个进程所需要处理的子矩阵的元素数量，该变量用于后续的数据分发*/
        local_len=n/my_sqrt(comm_sz);                              /*每个子矩阵的边长/行数/列数*/
        local_mat=(double *)malloc(local_sz*sizeof(double));       /*创建当前进程所需要处理的子矩阵的空间*/
        local_vec=(double *)malloc(local_len*sizeof(double));      /*创建当前进程所需要处理的向量分块的空间*/
        /*进程0向其他进程发送各个矩阵分块*/
        MPI_Scatter(mat,local_sz,MPI_DOUBLE,local_mat,local_sz,MPI_DOUBLE,0,MPI_COMM_WORLD);   
        /*制作各个进程所需要的向量元素分发表*/
        int *sendcounts=(int *)malloc(comm_sz*sizeof(int));
        int *displs=(int *)malloc(comm_sz*sizeof(int));
        for(i=0;i<comm_sz;i++)
        {
            sendcounts[i]=local_len;
            displs[i]=local_len*(i%my_sqrt(comm_sz));              /*单个矩阵分块所需要的向量元素个数为n/my_sqrt(comm_sz)，按照分块矩阵属于第几个列范围来分配向量元素*/
        }
        MPI_Scatterv(vec,sendcounts,displs,MPI_DOUBLE,local_vec,local_len,MPI_DOUBLE,0,MPI_COMM_WORLD);      /*进程0向其他进程发送各个向量元素*/
    }

    /*2.其他进程从进程0接收当前进程需要处理的块*/
    else 
    {
        MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);                  /*进程0首先向其他进程广播矩阵和向量的维数*/
        local_sz=n*n/comm_sz;                                      /*每一个进程所需要处理的分块矩阵的元素数量，该变量用于后续的数据分发*/
        local_len=n/my_sqrt(comm_sz);                              /*每个矩阵分块的边长/行数/列数*/
        local_mat=(double *)malloc(local_sz*sizeof(double));       /*创建当前进程所需要处理的矩阵分块的空间*/
        local_vec=(double *)malloc(local_len*sizeof(double));      /*创建当前进程所需要处理的向量元素的空间*/
        /*进程0向其他进程发送各个矩阵分块*/
        MPI_Scatter(mat,local_sz,MPI_DOUBLE,local_mat,local_sz,MPI_DOUBLE,0,MPI_COMM_WORLD);   
        /*制作各个进程所需要的向量元素分发表*/
        int *sendcounts=(int *)malloc(comm_sz*sizeof(int));
        int *displs=(int *)malloc(comm_sz*sizeof(int));
        for(i=0;i<comm_sz;i++)
        {
            sendcounts[i]=local_len;
            displs[i]=local_len*(i%my_sqrt(comm_sz));              /*单个矩阵分块所需要的向量元素个数为n/my_sqrt(comm_sz)，按照分块矩阵属于第几个列范围来分配向量元素*/
        }
        MPI_Scatterv(vec,sendcounts,displs,MPI_DOUBLE,local_vec,local_len,MPI_DOUBLE,0,MPI_COMM_WORLD);      /*进程0向其他进程发送各个向量元素*/
    }

    local_len=n/my_sqrt(comm_sz);    /*重新确认设置local_len*/

    /*3.各个进程计算自己获得的矩阵和向量的积*/
    local_res=(double *)malloc(local_len*sizeof(double));       /*单个进程的分块矩阵和分块向量相乘的结果*/
    for(i=0;i<local_len;i++)     /*分块矩阵的每一行与分块向量相乘*/
    {
        double temp=0;
        for(k=0;k<local_len;k++)
        {
            temp+=local_mat[i*local_len+k]*local_vec[k];
        }
        local_res[i]=temp;
    }
	

    /*4.相同行的矩阵块计算结果进行相加*/
    int local_num=my_sqrt(comm_sz);       /*矩阵一条边的分块划分的进程数量*/
    if(my_rank%local_num==0)              /*每行的第一个矩阵分块对应的矩阵收集同行其他进程的计算结果*/
    {
        local_temp=(double *)malloc(local_len*sizeof(double));      /*注意local_temp为接收的临时变量*/
        for(i=my_rank+1;i<my_rank+local_num;i++)
        {
            MPI_Recv(local_temp,local_len,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            for(k=0;k<local_len;k++)
            {
                local_res[k]+=local_temp[k];
            }
        }
    }
    else
    {
        MPI_Send(local_res,local_len,MPI_DOUBLE,my_rank-my_rank%local_num,0,MPI_COMM_WORLD);
    }

    /*5.将每行第一个矩阵分块的计算结果汇总收集到进程0*/
    if(my_rank%local_num==0&&my_rank!=0)
    {
        MPI_Send(local_res,local_len,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
    }
    else if(my_rank==0)
    {
        local_sum=(double *)malloc(n*sizeof(double));
        for(i=0;i<local_len;i++)
        {
            local_sum[i]=local_res[i];
        }
        for(i=0+local_num,p=1;i<comm_sz;i+=local_num,p++)
        {
            MPI_Recv(&local_sum[p*local_len],local_len,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);   /*注意这里不是local_sum[i*local_len]，否则会出错*/
        }
        for(i=0;i<n;i++)
        {
            printf("%lf ",local_sum[i]);
        }
        printf("\n");
    }

    if(local_mat!=NULL)
    free(local_mat);
    if(local_vec!=NULL)
    free(local_vec);
    if(local_sum!=NULL)
    free(local_sum);
    if(local_temp!=NULL)
    free(local_temp);
    if(mat!=NULL)
    free(mat);
    if(initmat!=NULL)
    {
        for(i=0;i<n;i++)
        {
            free(initmat[i]);
        }
        free(initmat);
    }
    if(local_res!=NULL)
    free(local_res);
    if(vec!=NULL)
    {
        free(vec);
    }

    MPI_Finalize();
    return 0;
}