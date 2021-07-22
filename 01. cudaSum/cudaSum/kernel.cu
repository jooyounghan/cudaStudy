#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <iostream>
#include <chrono>

class timer
{
private:
	std::chrono::system_clock::time_point start;
	std::chrono::system_clock::time_point end;
public : 
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

__global__ void cudaSum(int* _a, int* _b, int* _c)
{
	int tNum = threadIdx.x;
	int dNum = blockIdx.x;
	int idx = dNum * blockDim.x + tNum;
	_c[idx] = _a[idx] + _b[idx];
	//printf("%d = %d + %d \n", _c[idx], _a[idx], _b[idx]);
}

int main()
{

	int _threadx = 3;
	int _dimx = 3;
	int num = _threadx * _dimx;
	int SIZE = sizeof(int) * num;

	int* cpu_a;
	int* cpu_b;
	int* cpu_c;
	int* cpu_d;

	int* d_a;
	int* d_b;
	int* d_c;

	cpu_a = new int[num];
	cpu_b = new int[num];
	cpu_c = new int[num];
	cpu_d = new int[num];

	for (int i = 0; i < num; ++i)
	{
		cpu_a[i] = std::rand() % 10;
		cpu_b[i] = std::rand() % 10;
	}

	timer GPU;
	GPU.checkStart();
	cudaMalloc(&d_a, SIZE);
	cudaMalloc(&d_b, SIZE);
	cudaMalloc(&d_c, SIZE);

	cudaMemcpy(d_a, cpu_a, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, cpu_b, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, cpu_c, SIZE, cudaMemcpyHostToDevice);

	cudaSum << <_dimx, _threadx >> > (d_a, d_b, d_c);
	cudaDeviceSynchronize(); // synchronization function
	cudaMemcpy(cpu_c, d_c, SIZE, cudaMemcpyDeviceToHost);
	GPU.checkEnd();

	timer CPU;
	CPU.checkStart();
	for (int i = 0; i < num; ++i)
	{
		cpu_d[i] = cpu_a[i] + cpu_b[i];
	}
	CPU.checkEnd();

	bool check = true;
	for (int i = 0; i < num; ++i)
	{
		if (cpu_d[i] != cpu_c[i]) { check = false; break; }
	}
	
	if (check) { std::cout << "Well Done" << std::endl; }
	else { std::cout << "BAD" << std::endl; }

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	// Release host memory
	delete[] cpu_a; delete[] cpu_b; delete[] cpu_c; delete[] cpu_d;

	std::cout << " GPU Time spent : ";
	GPU.elasped();
	std::cout << " secs" << std::endl;
	std::cout << " CPU Time spent : ";
	CPU.elasped();
	std::cout << " secs" << std::endl;

}