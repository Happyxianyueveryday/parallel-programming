
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//Kernel方法matVecMulti:该方法依次将一个5000*5000矩阵中的第i列和一个5000维向量中的第i个元素相乘，相乘结果保存为结果矩阵res的第i列
//参数列表:mat--原始输入矩阵，vec--原始输入向量，res--结果矩阵
//附注:结果矩阵res必须初始化为0
__global__ void matVecMulti(const double mat[5000*5000],const double vec[5000],double res[5000*5000])
{
	//1. 计算进程编号
	int x = blockIdx.x*blockDim.x + threadIdx.x;    //进程列编号x
	int y = blockIdx.y*blockDim.y + threadIdx.y;    //进程行编号y

	//2. 假设进程列编号为x，行编号为y，则该进程计算原矩阵的第x列的第y*10行到第(y+1)*10-1行的元素与vec中的对应第x行元素相乘，再加到结果向量res上
	for (int i = y * 10; i < (y + 1) * 10; i++)
	{
		res[i*5000+x] += vec[x] * mat[i*5000+x];
	}
}

//Kernel方法vecSum:该方法依次将一个5000*5000矩阵中的第i列相加得到一个5000维的和向量
//参数列表:mat--加数向量组成的矩阵，res--结果向量
//附注:结果矩阵res必须初始化为0
__global__ void vecSum(const double vec[5000*5000],double res[5000])
{
	//1. 计算进程编号
	int x = blockIdx.x*blockDim.x + threadIdx.x;    //进程列编号x
	int y = blockIdx.y*blockDim.y + threadIdx.y;    //进程行编号y

	//2. 假设进程列编号为x，行编号为y，则该进程计算向量的第x*500+y行元素的和
	for (int i = 0; i < 5000; i++)
	{
		int row = x * 500 + y;
		res[x * 500 + y] += vec[row * 5000 + i];
	}
}

int main()
{
	//1. 设定线程结构
	dim3 gridDim(5000, 1);      //一个Grid含有1*5000个block，行数为1，列数为10
	dim3 blockDim(1, 500);    //一个block含有个500*1个thread，行数为500，列数为1
	dim3 gridDim2(10, 1);           //一个grid含有1*10个block，行数为1，列数为10
	dim3 blockDim2(1, 500);         //一个block含有个500*1个thread，行数为500，列数为1

	//2. 初始化host端矩阵host_A，向量host_x和结果向量host_res
	double *host_A = (double *)malloc(5000 * 5000 *sizeof(double));
	double *host_x = (double *)malloc(5000 * sizeof(double));
	double *host_res = (double *)malloc(5000 * 5000 * sizeof(double));
	double *host_final_res = (double *)malloc(5000 * sizeof(double));
	for (int i = 0; i < 5000; i++)
	{
		for (int j = 0; j < 5000; j++)
		{
			host_A[i * 5000 + j] = i - 0.1*j + 1;
		}
	}
	for (int i = 0; i < 5000; i++)
	{
		host_x[i] = 0.2*i - 0.1*sqrt(i);
	}
	memset(host_res, 0, 5000 * 5000 * sizeof(double));
	memset(host_final_res, 0, 5000 * sizeof(double));

	//3. 将host端的数据拷贝到device端
	double *device_A, *device_x, *device_res, *device_final_res;
	cudaMalloc((void **) &device_A, 5000 * 5000 * sizeof(double));
	cudaMalloc((void **) &device_x, 5000 * sizeof(double));
	cudaMalloc((void **) &device_res, 5000 * 5000 * sizeof(double));
	cudaMalloc((void **) &device_final_res, 5000 * sizeof(double));
	cudaMemcpy(device_A, host_A, 5000 * 5000 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(device_x, host_x, 5000 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(device_res, host_res, 5000 * 5000 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(device_final_res, host_final_res, 5000 * sizeof(double), cudaMemcpyHostToDevice);

	//4. 调用Kernel函数求解运算结果
	matVecMulti << <gridDim, blockDim >>> (device_A, device_x, device_res);
	vecSum << <gridDim2, blockDim2 >> > (device_res,device_final_res);

	//5. 将运算结果由device拷贝回host端
	cudaMemcpy(host_final_res, device_final_res, 5000 * sizeof(double), cudaMemcpyDeviceToHost);

	//6. 输出运算结果
	printf("final result:\n");
	for (int i = 0; i < 5000; i++)
	{
		if(i==0)
		printf("[%lf", host_final_res[i]);
		else if(i==4999)
		printf(", %lf]", host_final_res[i]);
		else
        printf(", %lf", host_final_res[i]);
	}

	return 0;
}
