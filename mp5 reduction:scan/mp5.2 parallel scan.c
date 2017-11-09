// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include <wb.h>

//*** Macros ***//
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


//*** Program-wide constants ***//
#define BLOCK_SIZE 1024


//*** Parallel scan kernels ***//
__global__ void scan(float *input, float *output, int len, int internalLayer) {

  //*** General Thread Info ***//

  int loadX;
  int loadStride;
  if (!internalLayer) {
    loadX = threadIdx.x + (blockIdx.x * blockDim.x * 2);
    loadStride = blockDim.x;
  } else {
    loadX = (threadIdx.x + 1) * (blockDim.x * 2) - 1;
    loadStride = blockDim.x * 2;
  }

  int storeX = threadIdx.x + (blockIdx.x * blockDim.x * 2);


  //*** Generate scanSegment ***//
  __shared__ float scanSegment[BLOCK_SIZE * 2];

  if (loadX < len)
    scanSegment[threadIdx.x] = input[loadX];
  else
    scanSegment[threadIdx.x] = 0;

  if (loadX + loadStride < len)
    scanSegment[threadIdx.x + blockDim.x] = input[loadX + loadStride];
  else
    scanSegment[threadIdx.x + blockDim.x] = 0;

  //** Parallel inclusive scan algorithm (based on Brent-Kung) **//
  // First scan half
  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();

    int index = (threadIdx.x + 1) * 2 * stride - 1;

    if (index < 2 * blockDim.x) {
      scanSegment[index] += scanSegment[index - stride];
    }
  }

  // Second scan half
  for (int stride = 2 * blockDim.x / 4; stride > 0; stride /= 2) {
    __syncthreads();

    int index = (threadIdx.x + 1) * 2 * stride - 1;

    if (index + stride < 2 * blockDim.x) {
      scanSegment[index + stride] += scanSegment[index];
    }
  }


  //*** Store partial scan result to output ***//
  __syncthreads();
  if (storeX < len)
    output[storeX] = scanSegment[threadIdx.x];
  if (storeX + blockDim.x < len)
    output[storeX + blockDim.x] = scanSegment[threadIdx.x + blockDim.x];
}

__global__ void add(float *input, float *output, float *sum, int len) {
  //*** General Thread Info ***//
  int x = threadIdx.x + (blockIdx.x * blockDim.x * 2);


  //*** Generate scanSegment ***//
  __shared__ float increment;
  if (threadIdx.x == 0)
    increment = blockIdx.x == 0 ? 0 : sum[blockIdx.x - 1];

  __syncthreads();


  //*** Store partial scan result to output ***//
  if (x < len)
    output[x] = input[x] + increment;
  if (x + blockDim.x < len)
    output[x + blockDim.x] = input[x + blockDim.x] + increment;
}



int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceScanBuffer;
  float *deviceScanSums;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);



  //*** Importing data and creating memory on host ***//
  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");



  wbLog(TRACE, "The number of input elements in the input is ", numElements);



  //*** Allocating GPU memory ***//
  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput,      numElements    * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanBuffer, numElements    * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanSums,   2 * BLOCK_SIZE * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput,     numElements    * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");



  //*** Clearing output memory ***//
  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");



  //*** Copying input memory to the GPU ***//
  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");



  //*** Performing CUDA computation ***//
  wbTime_start(Compute, "Performing CUDA computation");
  int numBlocks = ceil(numElements/float(BLOCK_SIZE * 2));

  dim3 dimGrid(numBlocks,   1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  scan<<<dimGrid, dimBlock>>>(
    deviceInput, deviceScanBuffer, numElements, 0
  );
  cudaDeviceSynchronize();

  dim3 singleGrid(1, 1, 1);
  scan<<<singleGrid, dimBlock>>>(
    deviceScanBuffer, deviceScanSums, numElements, 1
  );
  cudaDeviceSynchronize();

  add<<<dimGrid, dimBlock>>>(
    deviceScanBuffer, deviceOutput, deviceScanSums, numElements
  );
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");



  //*** Copying output memory to the CPU ***//
  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");




  //*** Freeing GPU Memory ***//
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceScanBuffer);
  cudaFree(deviceScanSums);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");



  //*** Check Solution ***//
  wbSolution(args, hostOutput, numElements);



  //*** Freeing CPU Memory ***//
  free(hostInput);
  free(hostOutput);



  //*** Exit ***//
  return 0;
}

