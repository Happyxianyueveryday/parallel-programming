#include <stdio.h>
#include <string.h>
#include <mpi.h>

/*习题3.1的第一部分：使用树形结构2(p68 图3-7)，且处理进程数量comm_sz为2的幂的情况*/
/*当进程数量comm_sz是2的幂时，假设进程数量为comm_sz，假设数组的总大小为total_sz*/
/*1.首先将数组中的total_size个元素分配到comm_sz个进程中，具体而言假设step=floor(total_sz/comm_sz)，则进程编号为0到comm_sz-2的进程每个处理的元素数量为step，具体而言，进程i处理的下标范围为[i*step,(i+1)*step)；最后一个进程处理的元素个数为total_sz-(comm_sz-1)*floor(total_sz/comm_sz)，其对应处理的范围为[i*step,total_sz)*/
/*2.然后，各个进程对它们所负责的数组元素范围进行串行求和*/
/*3.然后进行进程通信，我们使用树形结构2，假设需要进行通信的进程为活跃进程，活跃进程的数目为:current_sz，初始值设为comm_sz*/
/*4.若当前的进程编号i在编号0到活跃进程数量减1之间，即在范围[0,current_sz)之间，则需要进一步执行下面的步骤5和6，若不在该范围内，则不需要执行步骤5和6，直接跳到步骤7*/
/*5.若当前的进程编号在范围[0,current_sz)的前半部分，即[0,current_sz/2)，则该进程在本次的循环中是接收进程，从进程编号为(i+current_sz/2)的进程接收信息*/
/*6.若当前的进程编号在范围[0,current_sz)的后半部分，即[current_sz/2,current_sz)，则该进程在本次的循环中是发送进程，向进程编号为(i-current_sz/2)的进程发送消息*/
/*7.在每一次循环结束后，[current_sz/2,current_sz)部分的进程无需再进行任何通信（包括接收或者发送），即活跃进程数量current_sz减半，因此执行更新current_sz=current_sz/2*/
/*8.执行上述循环，直到活跃进程数为1为止，这时唯一的活跃进程——进程0中的值就是最终的求和结果*/

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
        /*首先判断当前进程是否是非活跃进程*/    /*这一部分非常关键，并不是所有的进程在每一次循环中都必须进行通信，例如通过这个if语句，第二次以及之后的循环中的进程7就不会被执行通信过程*/
        if(my_rank>=current_sz)    
        break;                               /*非活跃进程在以后的循环中已经无需再进行任何通信，故直接退出循环*/  /*例如进程7，在第二次循环中就会触发该if语句，之后就直接跳出循环，后续不会被执行通信过程，在本次实验的几个程序设计中，我们正是通过类似这种if语句来控制不同进程执行不同次数以及不同种类的通信过程*/
        /*不触发上述if语句的即为活跃进程，我们进一步确定该活跃进程应该执行发送还是接收操作*/
        else
        {
            if(my_rank>=0&&my_rank<current_sz/2)                 /*由p68 图3-7可知，编号在区间[0,current_sz/2)的为接收进程，接收来自进程i+current_sz/2的信息*/
            {
                MPI_Recv(&receive_res,1,MPI_DOUBLE,my_rank+current_sz/2,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);   /*接收进程通信信息*/
                add+=receive_res;
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