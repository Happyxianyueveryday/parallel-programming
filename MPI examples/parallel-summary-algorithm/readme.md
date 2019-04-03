# 并行求和算法

## 0. 源代码

| 源代码文件 | 使用的并行求和算法    |
|-----------|---------------------------------|
| 3.3.1.c | 使用树形结构2，进程数为2的幂    |
| 3.3.2.c | 使用树形结构1，进程数为2的幂 |
| 3.3.3.c | 使用树形结构2，进程数为任意整数 |
| 3.3.4.c | 使用树形结构1，进程数为任意整数 |
| 3.4.1.c | 使用蝶形结构，进程数为2的幂 |
| 3.4.2.c | 使用蝶形结构，进程数为任意整数 |

树形结构1，树形结构2，蝶形结构求和算法的基本图示如下。

+ 树形结构1求和算法图示
![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/parallel-summary-algorithm/pics/%E6%A0%91%E5%BD%A2%E7%BB%93%E6%9E%841.jpg)

+ 树形结构2求和算法图示
![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/parallel-summary-algorithm/pics/%E6%A0%91%E5%BD%A2%E7%BB%93%E6%9E%842.jpg)

+ 蝶形结构求和算法图示
![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/parallel-summary-algorithm/pics/%E8%9D%B6%E5%BD%A2%E7%BB%93%E6%9E%84.jpg)

## 2. 算法实现思路简介

### 3.3.1.c: 使用树形结构2，进程数为2的幂
  首先我们分析一下树形结构2的示意图
  
  ![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/parallel-summary-algorithm/pics/%E6%A0%91%E5%BD%A2%E7%BB%93%E6%9E%842.jpg)
  
  通过上述示意图，可以观察到以下几个性质：
  + 假设我们将在接下来的步骤中还需要进行进程间通信的进程称为活跃进程，则在每一轮通信活动完成后，活跃进程的数量减半。。
  + 在每一轮进程间通信活动中，每一个进程要么向其他进程发送自己的部分和，要么从别的进程接收部分和并与自己求解出的部分和相加。具体而言，在活跃进程中的前一半作为接收进程，后一半作为发送进程。
  
  因此实现算法为：
  + 步骤1：假设进程总数为comm_sz，活跃进程数量为current_sz，当前进程编号为my_rank，则首先将活跃进程数量作初始化：current_sz=comm_sz  
  
  + 步骤2：首先判断活跃进程的数量是否为1，若活跃进程数量current_sz==1，则这时说明所有其他进程的部分和均已汇总到进程0，因此直接跳转至步骤6。否则，即current_sz!=1，循环执行步骤3，步骤4和步骤5，直到满足current_sz==1时退出循环。
  
  + 步骤3：对于当前进程my_rank，首先判断当前进程是否为活跃进程，若当前进程的编号大于或等于活跃进程数量，即my_rank>=current_sz，则当前进程不是活跃进程，该进程无需通信，直接退出循环。否则，即my_rank<current_sz，这时当前进程是活跃进程，执行步骤4.  
  
  + 步骤4：由步骤3已知当前进程为活跃进程，我们进一步判断当前进程需要发送消息还是接收消息，由之前的分析可知，在活跃进程中前半的进程接收消息，而后半的进程发送消息。因此，当前进程编号my_rank若在范围\[0, current_sz/2)内，则是接收进程，接收来自进程(my_rank+current_sz/2)的部分和并与自己的部分和相加；若在范围\[current_sz/2, current_sz)内，则是发送进程，向进程(my_rank-current_sz/2)发送部分和。
  
  + 步骤5：在一轮通信结束时，活跃进程数减半，即current_sz=current_sz/2。
  
  + 步骤6：步骤2到步骤5的迭代结束后，由进程0输出全局总和。

  
  


