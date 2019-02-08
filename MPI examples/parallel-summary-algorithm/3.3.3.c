#include <stdio.h>
#include <string.h>
#include <mpi.h>

/*习题3.1的第三部分：使用树形结构2(p68 图3-7)，且处理进程数量comm_sz不一定为2的幂的情况*/
/*当进程数量comm_sz是2的幂时，假设进程数量为comm_sz，假设数组的总大小为total_sz*/
/*1.首先将数组中的total_size个元素分配到comm_sz个进程中，具体而言假设step=floor(total_sz/comm_sz)，则进程编号为0到comm_sz-2的进程每个处理的元素数量为step，具体而言，进程i处理的下标范围为[i*step,(i+1)*step)；最后一个进程处理的元素个数为total_sz-(comm_sz-1)*floor(total_sz/comm_sz)，其对应处理的范围为[i*step,total_sz)*/
/*2.然后，各个进程对它们所负责的数组元素范围进行串行求和*/
/*3.然后进行进程通信，我们使用树形结构2，假设需要进行通信的进程为活跃进程，活跃进程的数目为:current_sz，初始值设为comm_sz*/
/*4.由于条件已经修改成为comm_sz不一定为2的幂，因此我们作如下处理：*/
/*5.首先判断当前进程是否是活跃进程，只要当前进程的编号my_rank的范围在[0,current_sz)内，则当前进程为活跃进程，否则为非活跃进程。若当前进程为活跃进程，则继续执行下面的步骤6，7，否则直接退出循环*/
/*6.然后，判断当前进程是发送进程还是接收进程：
    (a)若当前的活跃进程数目current_sz是奇数：
    则编号在范围[0,current/2)的进程为接收进程，编号在范围[current/2,current_sz-1)的进程为发送进程，若为接收进程，则接收来自进程(my_rank+current_sz/2)的消息；若为发送进程，则将信息发送给进程(my_rank-current_sz/2)。特别的，编号为current_sz-1的进程将它的节点值发送给进程0，进程0除了接收进程(my_rank+current_sz/2)的消息外，还需要从编号为current_sz-1进程接收信息
    (b)若当前的活跃进程数目current_sz是偶数：
    则编号在范围[0,current/2)的进程为接收进程，编号在范围[current/2,current_sz)的进程为发送进程，若为接收进程，则接收来自进程(my_rank+current_sz/2)的消息；若为发送进程，则将信息发送给进程(my_rank-current_sz/2)*/
/*7.将活跃进程数减半，即current_sz=current_sz/2，不论current_sz原来是奇数或者偶数，都使用该式进行减半*/
/*8.当减半至cuurent_sz=1时，退出循环，这时进程0的值就是树形求和的结果*/


int main(void)
{
    int comm_sz=8;         /*本并行程序所使用的进程数量（变量）*/
    int my_rank;           /*当前进程的编号*/
    double lis[]={5,2,-1,-3,6,5,-7,2};   /*待求和的数组*/    
    int total_sz=8;        /*数组总大小*/

    /*创建MPI进程并获取当前的进程编号*/
    MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);     /*指定进程数量*/
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);     /*获取进程编号*/
    
    /*每一个进程首先先计算出需要计算的那一部分数组的范围*/
    int step=total_sz/comm_sz;
    int begin=my_rank*step;
    int end=(my_rank==comm_sz-1)?total_sz:(my_rank+1)*step;
    int i=0;
    double add=0;       /*当前进程所负责的数组范围内的元素求和结果*/
    for(i=begin;i<end;i++)
    {
        add+=lis[i];
    }
    
    /*进行进程通信*/
    /*此处执行进程通信时，每个进程初始时仅仅已知进程总数和自己的进程编号，这与串行编程有很大不同，仅能站在进程的角度，仅仅根据这两个已知量来设计通信过程，而不能站在上帝视角*/
    int current_sz=comm_sz;                  /*当前活跃的进程数，需要注意观察到的是：活跃进程在一次循环/通信中仅执行接收/发送操作中的一个*/
    double receive_res=0;                    /*来自发送进程的结果，必须注意初始值必须设定为0！*/
    while(current_sz!=1)
    {
        /*首先，判断当前进程是否为活跃进程*/
        if(my_rank>=current_sz)     
        break;        /*非活跃进程直接退出循环，因为不需要再进行通信*/
        /*然后，判断当前的活跃进程数为奇数还是偶数，对于奇数和偶数的情况，我们使用的算法略有不同*/
        else if(current_sz%2==1)     /*若当前的活跃进程数为奇数*/
        {
            /*最后，判断当前进程是发送进程还是接收进程*/
            if(my_rank==0)   /*特殊判断：0号进程*/
            {
                MPI_Recv(&receive_res,1,MPI_DOUBLE,my_rank+current_sz/2,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);   /*0号进程首先接收来自进程(my_rank+current_sz/2)的信息*/
                add+=receive_res;
                receive_res=0;   /*重置接收信息变量receive_res*/
                MPI_Recv(&receive_res,1,MPI_DOUBLE,current_sz-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);    /*0号进程随后接收来自进程(current_sz-1)的信息*/
                add+=receive_res;
            }
            else if(my_rank==current_sz-1)    /*特殊判断：最后剩下的无配对进程*/
            {
                MPI_Send(&add,1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
            }
            else if(my_rank>=0&&my_rank<current_sz/2)   /*其他接收进程*/
            {
                MPI_Recv(&receive_res,1,MPI_DOUBLE,my_rank+current_sz/2,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);   /*接收进程通信信息*/
                add+=receive_res;           /*将接收的通信信息和原先的计算结果相加*/
                receive_res=0;
            }
            else if(my_rank>=current_sz/2&&my_rank<current_sz-1)   /*其他发送进程*/
            {
                MPI_Send(&add,1,MPI_DOUBLE,my_rank-current_sz/2,0,MPI_COMM_WORLD);
            }
        }
        else if(current_sz%2==0)
        {
            /*最后，判断当前进程是发送进程还是接收进程*/
            if(my_rank>=0&&my_rank<current_sz/2)                 /*由p68 图3-7可知，编号在区间[0,current_sz/2)的为接收进程，接收来自进程i+current_sz/2的信息*/
            {
                MPI_Recv(&receive_res,1,MPI_DOUBLE,my_rank+current_sz/2,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);   /*接收进程通信信息*/
                add+=receive_res;           /*将接收的通信信息和原先的计算结果相加*/
                receive_res=0;
            }
            else if(my_rank>=current_sz/2&&my_rank<current_sz)   /*由p68 图3-7可知，编号在区间[current_sz/2,current_sz)的为接收进程，发送给进程i-current_sz/2对应的信息*/
            {
                MPI_Send(&add,1,MPI_DOUBLE,my_rank-current_sz/2,0,MPI_COMM_WORLD);
            }            
        }
        current_sz=current_sz/2;
    }

    /*输出运行结果*/
    /*由进程0输出运行结果*/
    if(my_rank==0)
    {
        printf("%lf",add);
    }
    MPI_Finalize();
}