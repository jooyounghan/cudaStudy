#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

//dim3 dimGrid(4, 1, 1);		//unit3 types that accesssible through x, y, and z fields
//dim3 dimBlock(8, 1, 1);		//unit3 types that accesssible through x, y, and z fields

// vecAdD<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);	//와 같은 형식으로 커널에 그리드와 블록을 지정해준다.

__global__ void checkDeviceIndex()
{
	printf("threadIdx : (%d, %d, %d) / blockIdx : (%d, %d, %d) / blockDim : (%d, %d, %d) / gridDim : (%d, %d, %d) \n"
		, threadIdx.x, threadIdx.y, threadIdx.z
		, blockIdx.x, blockIdx.y, blockIdx.z
		, blockDim.x, blockDim.y, blockDim.z
		, gridDim.x, gridDim.y, gridDim.z);
}


int main()
{
	dim3 dimGrid(4, 1, 1);
	dim3 dimBlock(8, 1, 1);	
	checkDeviceIndex << <dimGrid, dimBlock >> > ();
	cudaDeviceReset();
	return 0;
}


