#include <stdio.h>
#include <string.h>
#include <mpi.h>

/*习题3.1的第四部分：使用树形结构1(p67 图3-6)，且处理进程数量不一定为comm_sz为2的幂的情况*/
/*当进程数量comm_sz是2的幂时，假设进程数量为comm_sz，假设数组的总大小为total_sz*/
/*1.首先将数组中的total_size个元素分配到comm_sz个进程中，具体而言假设step=floor(total_sz/comm_sz)，则进程编号为0到comm_sz-2的进程每个处理的元素数量为step，具体而言，进程i处理的下标范围为[i*step,(i+1)*step)；最后一个进程处理的元素个数为total_sz-(comm_sz-1)*floor(total_sz/comm_sz)，其对应处理的范围为[i*step,total_sz)*/
/*2.然后，各个进程对它们所负责的数组元素范围进行串行求和*/
/*3.然后进行进程通信，我们使用树形结构1，在一个循环中，我们假设需要进行通信的进程为活跃进程，不需要再继续进行任何通信的进程为非活跃进程，我们再初始化一个步长变量step=1
/*4.首先，我们确定哪些进程是活跃进程，我们仅知道当前的进程编号为my_rank，总进程数量为comm_sz，若编号my_rank可以表示为0+step*k (其中k=0,1,2,...)(也即i能被步长step整除)则为活跃进程，否则是非活跃进程，若为非活跃进程，则直接跳到步骤6，否则继续执行下面的步骤5
/*5.然后，这时我们已经直到当前的进程my_rank为活跃进程，我们进一步确认当前进程my_rank是发送进程还是接收进程，这一步上相对于进程数为的2的幂的情况需要作出修改
    (1)首先我们计算当前的活跃进程数量，直接使用进程数量comm_sz整除以步长step即可得到当前的活跃进程数量current_sz
    (2)若当前活跃进程数量current_sz为奇数，则若当前进程编号my_rank除以步长step为奇数，则该进程是发送进程，将信息发送到进程(my-rank-step)；则若当前进程编号my_rank除以步长step为偶数，则该进程是接收进程，从进程(my-rank+step)接收信息；特别的，若当前进程编号my_rank除以步长step的值等于current_sz-1，则当前进程就是最后一个无法配对的进程，该进程需要将其信息发给进程0，进程0这时也要特殊处理，接收来自这个无法匹配的进程的信息，这些不能配对的进程一旦发送完自己的信息，就立刻退出接下来所需要进行的一切通信过程
    (3)若当前活跃进程数量current_sz为偶数，则若当前进程编号my_rank除以步长step为奇数，则该进程是发送进程，将信息发送到进程(my-rank-step)；则若当前进程编号my_rank除以步长step为偶数，则该进程是接收进程，从进程(my-rank+step)接收信息
*/

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
    int stepa=total_sz/comm_sz;
    int begin=my_rank*stepa;
    int end=(my_rank==comm_sz-1)?total_sz:(my_rank+1)*stepa;
    int i=0;
    double add=0;       /*当前进程所负责的数组范围内的元素求和结果*/
    for(i=begin;i<end;i++)
    {
        add+=lis[i];
    }
    
    /*进行进程通信*/
    /*此处执行进程通信时，每个进程初始时仅仅已知进程总数和自己的进程编号，这与串行编程有很大不同，仅能站在进程的角度，仅仅根据这两个已知量来设计通信过程，而不能站在上帝视角*/
    int step=1;             /*步长，步长的初始值为1*/
    double receive_res=0;   /*从其他进程接收到的结果，初始值必须设为0！*/
    double receive_res_2=0;
    while((comm_sz/step)!=1)     /*步长大于等于comm_sz，则通信过程终止*/
    {
        int current_sz=comm_sz/step;
        if(current_sz%2) /*活跃进程数为奇数的情况，这时处理方法和进程数量是2的幂时不同*/
        {
            /*首先判断当前进程是否是活跃进程，若当前进程编号my_rank能够整除以步长step则为活跃进程*/
            if(my_rank%step==0)
            {
                if(my_rank==0)    /*特殊处理：进程0*/
                {
                    MPI_Recv(&receive_res,1,MPI_DOUBLE,my_rank+step,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);   /*先接收进程my_rank+step的信息*/
                    add+=receive_res;
                    receive_res=0;
                    MPI_Recv(&receive_res_2,1,MPI_DOUBLE,step*(current_sz-1),0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);   /*再接收进程(step*(current_sz-1))的信息*/
                    add+=receive_res_2;
                    receive_res_2=0;
                }
                else if(my_rank/step==current_sz-1)    /*最后一个无法匹配的进程*/
                {
                    MPI_Send(&add,1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
                    break;                             /*这一步非常关键，一旦因为无法匹配而向进程0发送了自己的节点值，则该节点值不再参与下面进行的任何通信*/
                }
                else if((my_rank/step)%2)    
                {
                    MPI_Send(&add,1,MPI_DOUBLE,my_rank-step,0,MPI_COMM_WORLD);
                }
                else
                {
                    MPI_Recv(&receive_res,1,MPI_DOUBLE,my_rank+step,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    add+=receive_res;
                    receive_res=0;   /*重置receive_res*/
                }
            }   
            else    /*其他情况为非活跃进程，这些进程无需再进行通信，因此可以直接退出循环*/
            {
                break;
            }
        }
        else    /*活跃进程为偶数的情况，这时处理方法和进程数量是2的幂时相同*/
        {
            /*首先判断当前进程是否是活跃进程，若当前进程编号my_rank能够整除以步长step则为活跃进程*/
            if(my_rank%step==0)
            {
                /*进一步判断当前进程是发送进程还是接收进程，若进程编号my_rank除以步长step的结果为偶数就是接收进程，接收来自进程(i+step)的信息；为奇数就是发送进程，发送信息给编号为(i-step)的进程*/
                int temp=my_rank/step;
                if(temp%2)    
                {
                    MPI_Send(&add,1,MPI_DOUBLE,my_rank-step,0,MPI_COMM_WORLD);
                }
                else    
                {
                    MPI_Recv(&receive_res,1,MPI_DOUBLE,my_rank+step,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    add+=receive_res;
                    receive_res=0;   /*重置receive_res*/
                }
            }   
            else    /*其他情况为非活跃进程，这些进程无需再进行通信，因此可以直接退出循环*/
            {
                break;
            }
        }
        step=step*2;
    }

    /*输出运行结果*/
    /*由进程0输出运行结果*/
    if(my_rank==0)
    {
        printf("%lf",add);
    }
    MPI_Finalize();
}