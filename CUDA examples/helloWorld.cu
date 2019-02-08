
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//kernel方法helloWorld: 该方法将各个线程输出的消息写入给定数组
//参数列表: output--含有size个bool元素的签到数组，其中size为进程总数量，签到数组对应位置的元素值为1则代表了该线程成功创建并执行了Kernel函数
__global__ void helloWorld(bool *output)
{
	//1. 计算进程编号
	int x = blockIdx.x*blockDim.x + threadIdx.x;    //进程列编号x
	int y = blockIdx.y*blockDim.y + threadIdx.y;    //进程行编号y
	int nx = gridDim.x*blockDim.x;   //进程列数
	int ny = gridDim.y*blockDim.y;   //进程行数

	//2. 将对应位置的bool数组元素设定为1表示当前进程进行了响应
	output[y*nx + x] = 1;
}

int main()
{
	//1. 设定线程结构
	dim3 gridDim(2, 4);      //一个Grid含有4*2个block，行数为4，列数为2
	dim3 blockDim(8, 16);    //一个block含有16*8个thread，行数为16，列数为8
	int nx = gridDim.x*blockDim.x;   //线程列数
	int ny = gridDim.y*blockDim.y;   //线程行数
	int thread_size = gridDim.x*blockDim.x*gridDim.y*blockDim.y;    //线程总数

	//2. 创建签到数组
	bool *host_output = (bool *)malloc(thread_size * sizeof(bool));
	bool *device_output;
	cudaMalloc((void **)&device_output, thread_size * sizeof(bool));

	//3. 调用kernel函数生成签到数组
	helloWorld<<< gridDim, blockDim >>> (device_output);

	//4. 将签到数组的信息由device拷贝至host
	cudaMemcpy(host_output, device_output, thread_size * sizeof(bool), cudaMemcpyDeviceToHost);

	
	//5. 获得的数组由host将输出信息数组写入到文件
	FILE *file = fopen("output.txt", "w");
	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			if (host_output[i*nx + j] == 1)
			{
				char a[100] = "Hello World from thread (";
				char b[100] = ",";
				char c[100] = ") in block (";
				char d[100] = ").\n";
				char y[10];     //线程在block中的行编号y
				sprintf(y, "%d", int(i%blockDim.y));
				char x[10];     //线程在block中的列编号x
				sprintf(x, "%d", int(j%blockDim.x));
				char gy[10];    //线程所在block的行编号y
				sprintf(gy, "%d", int(i/blockDim.y));
				char gx[10];    //线程所在block的列编号y
				sprintf(gx, "%d", int(j/blockDim.x));
				strcat(a, y);
				strcat(a, b);
				strcat(a, x);
				strcat(a, c);
				strcat(a, gy);
				strcat(a, b);
				strcat(a, gx);
				strcat(a, d);
				fputs(a, file);
			}
		}
	}

	return 0;
}
