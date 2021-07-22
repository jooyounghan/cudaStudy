
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <iostream>
#include <chrono>

/*                 DYNAMIC MATRIX                 */
// 동적 2차 어레이의 경우, 데이터가 Row마다 연속적으로 할당되고
// 다른 Row와의 연속성을 보장 받을 수 없기 때문에 1차원으로 변환하여 풀어주어야한다.

#define ROW_SIZE (32)
#define K_SIZE   (128)
#define COL_SIZE (32)
#define WORK_LOAD (10240)

class timer
{
private:
	std::chrono::system_clock::time_point start;
	std::chrono::system_clock::time_point end;
public:
	void checkStart()
	{
		start = std::chrono::system_clock::now();
	}
	void checkEnd()
	{
		end = std::chrono::system_clock::now();
	}
	void elasped()
	{
		std::cout << std::chrono::duration<double>(end - start).count();
	}
	void checkTime()
	{
		std::cout << std::chrono::duration<double>(std::chrono::system_clock::now() - start).count() << std::endl;
	}

	std::chrono::system_clock::rep getElasped()
	{
		return (end - start).count();
	}

};

__global__ void matMul(int* _kernelResult, int* _kernelInput1, int* _kernelInput2)
{


	int row = threadIdx.y;
	int	col = threadIdx.x;
	int idx = blockDim.x * row + col;
	_kernelResult[idx] = 0;
	for (int k = 0; k < K_SIZE; k += 1)
	{
		for (int l = 0; l < WORK_LOAD; l += 1)
		{
			_kernelResult[idx] += _kernelInput1[K_SIZE * row + k] * _kernelInput2[COL_SIZE * k + col];
		}
	}
}



int main()
{
	// 행렬 연산 A X B = C
	// 호스트에서 메모리와 초기 데이터를 설정해주는 단계
	int* testInput1 = new int [ROW_SIZE * K_SIZE];			// A (2차원을 1차원으로 변환)
	int* testInput2 = new int [K_SIZE * COL_SIZE];			// B
	int* device_result = new int [ROW_SIZE * COL_SIZE];		// C (device에서 처리된 값이 들어올 녀석)
	int* host_result = new int [ROW_SIZE * COL_SIZE];		// C (host)
	
	for (int idx = 0; idx < ROW_SIZE * K_SIZE; idx += 1)
	{
		testInput1[idx] = rand() % 10;
	}
	for (int idx = 0; idx < K_SIZE * COL_SIZE; idx += 1)
	{
		testInput2[idx] = rand() % 10;
	}

	timer CPU;
	CPU.checkStart();
	for (int r = 0; r < ROW_SIZE; r += 1)
	{
		for (int c = 0; c < COL_SIZE; c += 1)
		{
			host_result[r * COL_SIZE + c] = 0;
			for (int k = 0; k < K_SIZE; k += 1)
			{
				for (int l = 0; l < WORK_LOAD; l += 1)
				{
					host_result[r * COL_SIZE + c] += testInput1[K_SIZE * r + k] * testInput2[COL_SIZE * k + c];
				}
			}
		}
	}
	CPU.checkEnd();


	// 디바이스에서 메모리를 설정해주는 단계

	timer GPU;
	GPU.checkStart();
	int* device_testInput1 = nullptr;
	int* device_testInput2 = nullptr;
	int* kernel_result = nullptr;


	cudaMalloc(&device_testInput1, sizeof(int) * ROW_SIZE * K_SIZE);
	cudaMalloc(&device_testInput2, sizeof(int) * K_SIZE * COL_SIZE);
	cudaMalloc(&kernel_result, sizeof(int) * ROW_SIZE * COL_SIZE);

	cudaMemcpy(device_testInput1, testInput1, sizeof(int) * ROW_SIZE * K_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(device_testInput2, testInput2, sizeof(int) * K_SIZE * COL_SIZE, cudaMemcpyHostToDevice);

	dim3 blockDim(COL_SIZE, ROW_SIZE);
	matMul << <1, blockDim >> > (kernel_result, device_testInput1, device_testInput2);

	cudaMemcpy(device_result, kernel_result, sizeof(int) * ROW_SIZE * COL_SIZE, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	GPU.checkEnd();
	//Checking!

	bool flag = true;
	for (int idx = 0; idx < ROW_SIZE * COL_SIZE; idx += 1)
	{
		if (host_result[idx] != device_result[idx])
		{
			flag = false;
			std::cout << "wrong...!!" << std::endl;
			break;
		}
	}
	if (flag) { std::cout << "Right!" << std::endl; }

	std::cout << " CPU Time spent : ";
	CPU.elasped();
	std::cout << " secs" << std::endl;
	std::cout << " GPU Time spent : ";
	GPU.elasped();
	std::cout << " secs" << std::endl;

	cudaFree(device_testInput1);
	cudaFree(device_testInput2);
	cudaFree(kernel_result);
	delete[] testInput1;
	delete[] testInput2;
	delete[] device_result;
	delete[] host_result;
} 