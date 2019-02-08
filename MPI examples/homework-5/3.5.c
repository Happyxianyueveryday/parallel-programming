#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(void)
{
    int comm_sz;      /*进程总数*/
    int my_rank;      /*当前进程编号*/
    int i,k,p;        /*计数变量*/
    int local_sz;     /*每个进程所需要计算的矩阵列数*/
    double *local_mat;    /*当前进程中接收到的矩阵块*/
    double *local_vec;    /*当前进程中接收到的向量块*/
    double *local_sum;    /*0号进程中的最终结果向量*/
    double *mat;          /*输入矩阵*/
    double *vec;          /*输入向量*/
    int n;                /*矩阵和向量的维数*/
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    /*我们将本实验分为以下几个主要步骤来完成*/
    /*1. 进程0读取向量输入和矩阵输入，并将矩阵进行按列分块，并使用块划分法分发到各个进程*/
    if(my_rank==0)
    {
        /*首先，进程0读取输入的方阵和向量大小，方阵的行数，列数以及向量维数相等*/
        printf("Enter the number of matrix rows/cols:\n");
        scanf("%d",&n);
        mat=(double *)malloc(n*n*sizeof(double));     /*输入矩阵，用一维数组模拟的二维数组，二维数组的每一行存储输入矩阵的每一列*/
        vec=(double *)malloc(n*sizeof(double));       /*输入向量*/

        /*然后，进程0读取输入的矩阵和向量*/
        printf("Enter the matrix:\n",n,n);
        for(i=0;i<n;i++)
        {
            for(k=0;k<n;k++)
            {
                scanf("%lf",&mat[k*n+i]);    /*注意，由于本实验中需要进行按列划分，因此二维数组的n行分别存储输入矩阵的n列*/
            }
        }
        printf("Enter the vector:\n",n);
        for(i=0;i<n;i++)
        {
            scanf("%lf",&vec[i]);     
        }
        /*然后，进程0将矩阵按列分块并且以块划分的划分方法分发到各个进程中，并且将向量的对应部分发送到各个进程中*/
        MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);                  /*进程0首先向其他进程广播矩阵和向量的维数*/
        local_sz=n/comm_sz;                                        /*每一个进程所需要处理的列数*/
        local_mat=(double *)malloc(local_sz*n*sizeof(double));     /*创建当前进程所需要处理的矩阵列元素的空间*/
        local_vec=(double *)malloc(local_sz*sizeof(double));       /*创建当前进程所需要处理的向量元素的空间*/
        local_sum=(double *)malloc(n*sizeof(double));              /*创建结果向量所需要的空间*/
        MPI_Scatter(mat,local_sz*n,MPI_DOUBLE,local_mat,local_sz*n,MPI_DOUBLE,0,MPI_COMM_WORLD);   /*进程0向其他进程发送各个矩阵列*/
        MPI_Scatter(vec,local_sz,MPI_DOUBLE,local_vec,local_sz,MPI_DOUBLE,0,MPI_COMM_WORLD);   /*进程0向其他进程发送各个向量元素*/
    }

    /*2.其他进程从进程0接收当前进程需要处理的块*/
    else 
    {
        MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);   /*首先接收自身所需要处理的块数目*/
        local_sz=n/comm_sz;                               /*每一个进程所需要处理的列数*/
        local_mat=(double *)malloc(local_sz*n*sizeof(double));  /*创建当前进程所需要处理的矩阵列元素的空间*/
        local_vec=(double *)malloc(local_sz*sizeof(double));    /*创建当前进程所需要处理的向量元素的空间*/
        MPI_Scatter(mat,local_sz*n,MPI_DOUBLE,local_mat,local_sz*n,MPI_DOUBLE,0,MPI_COMM_WORLD);  
        MPI_Scatter(vec,local_sz,MPI_DOUBLE,local_vec,local_sz,MPI_DOUBLE,0,MPI_COMM_WORLD);   
    }

    /*3.每个进程将向量元素乘以矩阵对应的列，然后将各个列相加*/
    for(i=0;i<local_sz;i++)
    {
        for(k=n*i;k<n*(i+1);k++)    /*必须注意此处每个向量元素乘以矩阵中的一个列，即n个元素，而不是local_sz个*/
        {
            local_mat[k]*=local_vec[i];
        }
    }
    for(k=n;k<local_sz*n;k++)
    {
        local_mat[k%n]+=local_mat[k];
    }
    
    /*4.最终每个进程将结果汇总相加，即得到最终运算结果*/
    MPI_Reduce(local_mat,local_sum,n,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    /*5.由进程0输出最终运算结果*/
    if(my_rank==0)
    {
        printf("The final result is:\n");
        for(i=0;i<n;i++)
        {
            printf("%lf ",local_sum[i]);
        }
        printf("\n");
    }

    free(local_mat);    /*当前进程中接收到的矩阵块*/
    free(local_vec);    /*当前进程中接收到的向量块*/

    if(my_rank==0)
    {
        free(mat);          /*输入矩阵*/
        free(vec);          /*输入向量*/
        free(local_sum);    /*0号进程中的最终结果向量*/
    }

    MPI_Finalize();
    return 0;
}