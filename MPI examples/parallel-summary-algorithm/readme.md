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

下面附上树形结构1，树形结构2，蝶形结构求和算法的基本图示。

+ 树形结构1求和算法图示
![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/parallel-summary-algorithm/pics/%E6%A0%91%E5%BD%A2%E7%BB%93%E6%9E%841.jpg)

+ 树形结构2求和算法图示
![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/parallel-summary-algorithm/pics/%E6%A0%91%E5%BD%A2%E7%BB%93%E6%9E%842.jpg)

+ 蝶形结构求和算法图示
![avatar](https://github.com/Happyxianyueveryday/parallel-programming/blob/master/MPI%20examples/parallel-summary-algorithm/pics/%E8%9D%B6%E5%BD%A2%E7%BB%93%E6%9E%84.jpg)


