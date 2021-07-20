
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <iostream>
#include <chrono>

/*                 DYNAMIC MATRIX                 */

// 동적 2차 어레이의 경우, 데이터가 Row마다 연속적으로 할당되고
// 다른 Row와의 연속성을 보장 받을 수 없기 때문에
// 
// cudaMalloc((void**)&ppArray_a, 10 * sizeof(int*));
//
// for (int i = 0; i < 10; i++)
//
// {
//
//     cudaMalloc(&someHostArray[i], 100 * sizeof(int)); /* Replace 100 with the dimension that u want */
//
// }
//
//cudaMemcpy(ppArray_a, someHostArray, 10 * sizeof(int*), cudaMemcpyHostToDevice);
// 와 같은 형태로 데이터를 고려해준다.

#define ROW_SIZE (32)
#define K_SIZE   (128)
#define COL_SIZE (32)
#define WORK_LOAD (1024)