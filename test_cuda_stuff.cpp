#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "cuda_yolo.h"
#define N 10

void test_cuda_stuff()
{
 int a[N], b[N], c[N];
 int *deva, *devb, *devc;

 cudaError_t status1 = cudaMalloc((void **)&deva, N*sizeof(int));
 check_error(status1);
 cudaError_t status2 = cudaMalloc((void **)&devb, N*sizeof(int));
 check_error(status2);
 cudaError_t status3 = cudaMalloc((void **)&devc, N*sizeof(int));
 check_error(status3);


 for (int i = 0; i < N; i++)
 {
  a[i] = -i;
  b[i] = i*i;
 }

 cudaError_t status4 = cudaMemcpy(deva, a, N*sizeof(int), cudaMemcpyHostToDevice);
 check_error(status4);
 cudaError_t status5 = cudaMemcpy(devb, b, N*sizeof(int), cudaMemcpyHostToDevice);
 check_error(status5);
 cudaError_t status6 = cudaMemcpy(devc, c, N*sizeof(int), cudaMemcpyHostToDevice);
 check_error(status6);
 // add << <N, 1 >> >(deva, devb, devc);

 // cudaMemcpy(c, devc, N*sizeof(int), cudaMemcpyDeviceToHost);
 // for (int i = 0; i < N; i++)
 // {
 //  printf("%d+%d=%d\n", a[i], b[i], c[i]);
 // }
 // cudaFree(deva);
 // cudaFree(devb);
 // cudaFree(devc);
}
