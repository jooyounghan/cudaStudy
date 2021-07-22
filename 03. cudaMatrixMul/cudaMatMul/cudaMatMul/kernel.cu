#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <iostream>
#include <chrono>

/*                 STATIC MATRIX                 */

// 정적 2차 어레이의 경우, 데이터가 연속적으로 할당되기 때문에,
// CUDA KERNAL 내에서 처리할 때에는, 1차원으로 고려하고 계산해 준다.

#define ROW_SIZE (32)
#define K_SIZE   (128)
#define COL_SIZE (32)
#define WORK_LOAD (1024)
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


__global__ void matMul(int* _matResult, int* _matInput1, int* _matInput2)
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    int idx = row * blockDim.x + col;
    _matResult[idx] = 0;
    for (int k = 0; k < K_SIZE; k += 1)
    {
        for (int l = 0; l < WORK_LOAD; l += 1)
        {
            _matResult[idx] += _matInput1[K_SIZE * row + k] * _matInput2[COL_SIZE * k + col];
        }
    }
}

__global__ void shared_matMul(int* _matResult, int* _matInput1, int* _matInput2)
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    int idx = row * blockDim.x + col;
    //shared_memory를 통하여 효율성을 높여보자.
    //shared memory의 경우, block이 공유하는 메모리이다.
    __shared__ int sInput1[ROW_SIZE][K_SIZE];       // 4bytes * ROW_SIZE(32) * K_SIZE(128) = 16KB
    __shared__ int sInput2[K_SIZE][COL_SIZE];       // 4bytes * ROW_SIZE(32) * K_SIZE(128) = 16KB

    for (int k = 0; k < K_SIZE; k += 1)
    {
        sInput1[row][k] = _matInput1[K_SIZE * row + k];
        sInput2[k][col] = _matInput2[COL_SIZE * k + col];
    }
    __syncthreads();    // 모든 thread가 shared memory에 데이터를 올려둘 때 까지 기다림


    _matResult[idx] = 0;
    for (int k = 0; k < K_SIZE; k += 1)
    {
        for (int l = 0; l < WORK_LOAD; l += 1)
        {
            _matResult[idx] += sInput1[row][k] * sInput2[k][col];
        }
    }
}

__global__ void shared_matMul_2(int* _matResult, int* _matInput1, int* _matInput2)
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    int idx = row * blockDim.x + col;

    //shared_memory를 통하여 효율성을 높여보자.
    __shared__ int sResult[ROW_SIZE][COL_SIZE];       // 4bytes * ROW_SIZE(32) * COL_SIZE(32) = 4KB

    for (int k = 0; k < K_SIZE; k += 1)
    {
        for (int l = 0; l < WORK_LOAD; l += 1)
        {
            sResult[row][col] += _matInput1[K_SIZE * row + k] * _matInput2[COL_SIZE * k + col];
        }
    }
    _matResult[idx] = sResult[row][col];
}

__global__ void register_matMul(int* _matResult, int* _matInput1, int* _matInput2)
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    int idx = row * blockDim.x + col;

    //shared_memory를 통하여 효율성을 높여보자.
    int sResult = 0;

    for (int k = 0; k < K_SIZE; k += 1)
    {
        for (int l = 0; l < WORK_LOAD; l += 1)
        {
            sResult += _matInput1[K_SIZE * row + k] * _matInput2[COL_SIZE * k + col];
        }
    }
    _matResult[idx] = sResult;
}

int main()
{
    int matInput1[ROW_SIZE][K_SIZE];	// m * k
    int matInput2[K_SIZE][COL_SIZE];	// k * n
    int hostResult[ROW_SIZE][COL_SIZE];	// host result
    int deviceResult[ROW_SIZE][COL_SIZE];	// device result

    for (int r = 0; r < ROW_SIZE; r += 1)
    {
        for (int k = 0; k < K_SIZE; k += 1)
        {
            matInput1[r][k] = rand() % 10;
        }
    }


    for (int k = 0; k < K_SIZE; k += 1)
    {
        for (int c = 0; c < COL_SIZE; c += 1)
        {
            matInput2[k][c] = rand() % 10;
        }
    }

    timer CPU;
    CPU.checkStart();
    for (int r = 0; r < ROW_SIZE; r++)
    {
        for (int c = 0; c < COL_SIZE; c++)
        {
            hostResult[r][c] = 0;
            for (int k = 0; k < K_SIZE; k++)
            {
                for (int l = 0; l < WORK_LOAD; l += 1)
                {
                    hostResult[r][c] += matInput1[r][k] * matInput2[k][c];
                }
            }
        }
    }
    CPU.checkEnd();

    timer GPU_memset;
    GPU_memset.checkStart();

    int* deviceMatInput1;
    int* deviceMatInput2;
    int* deviceMatResult;
    deviceMatInput1 = nullptr;
    deviceMatInput2 = nullptr;
    deviceMatResult = nullptr;

    cudaMalloc(&deviceMatInput1, sizeof(int) * ROW_SIZE * K_SIZE);
    cudaMalloc(&deviceMatInput2, sizeof(int) * COL_SIZE * K_SIZE);
    cudaMalloc(&deviceMatResult, sizeof(int) * ROW_SIZE * COL_SIZE);


    cudaMemcpy(deviceMatInput1, matInput1, sizeof(float) * ROW_SIZE * K_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatInput2, matInput2, sizeof(float) * COL_SIZE * K_SIZE, cudaMemcpyHostToDevice);
    GPU_memset.checkEnd();
    dim3 blockDim(COL_SIZE, ROW_SIZE);

    timer GPU;
    GPU.checkStart();
    matMul <<<1, blockDim>>>(deviceMatResult, deviceMatInput1, deviceMatInput2);
    cudaDeviceSynchronize();
    GPU.checkEnd();

    timer GPU_shared;
    GPU_shared.checkStart();
    shared_matMul << <1, blockDim >> > (deviceMatResult, deviceMatInput1, deviceMatInput2);
    cudaDeviceSynchronize();
    GPU_shared.checkEnd();

    timer GPU_shared2;
    GPU_shared2.checkStart();
    shared_matMul_2 << <1, blockDim >> > (deviceMatResult, deviceMatInput1, deviceMatInput2);
    cudaDeviceSynchronize();
    GPU_shared2.checkEnd();

    timer GPU_register;
    GPU_register.checkStart();
    register_matMul << <1, blockDim >> > (deviceMatResult, deviceMatInput1, deviceMatInput2);
    cudaDeviceSynchronize();
    GPU_register.checkEnd();

    timer GPU_memcpy;
    GPU_memcpy.checkStart();
    cudaMemcpy(deviceResult, deviceMatResult, sizeof(float) * ROW_SIZE * COL_SIZE, cudaMemcpyDeviceToHost);
    GPU_memcpy.checkEnd();


    bool flag = true;
    for (int r = 0; r < ROW_SIZE; r++)
    {
        if (flag == false) { break; }
        for (int c = 0; c < COL_SIZE; c++)
        {
            if (hostResult[r][c] != deviceResult[r][c]) { flag = false; }
        }
    }

    if (flag) { std::cout << "Well Done" << std::endl; }
    else { std::cout << "Wrong..." << std::endl; }

    cudaFree(deviceMatInput1);
    cudaFree(deviceMatInput2);
    cudaFree(deviceMatResult);

    std::cout << "CPU Time spent : ";
    CPU.elasped();
    std::cout << " secs" << std::endl;
    std::cout << "GPU Time spent \t\t";
    std::cout << "GPU_shared(32KB) Time spent \t\t";
    std::cout << "GPU_shared(20KB) Time spent";
    std::cout << std::endl;
    GPU.elasped();
    std::cout << " secs";
    std::cout << "\t\t";
    GPU_shared.elasped();
    std::cout << " secs";
    std::cout << "\t\t\t\t";
    GPU_shared2.elasped();
    std::cout << " secs" << std::endl;

    std::cout << "GPU_register Time spent : ";
    GPU_register.elasped();
    std::cout << " secs" << std::endl;

    std::cout << "GPU Memory Setting Time spent : ";
    GPU_memset.elasped();
    std::cout << " secs" << std::endl;
    std::cout << "GPU Memory Copy from Device to Host Time spent : ";
    GPU_memcpy.elasped();
    std::cout << " secs" << std::endl;

}
