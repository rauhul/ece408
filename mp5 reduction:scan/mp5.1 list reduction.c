// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

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

//*** Sum Reduction kernel ***//
__global__ void total(float *input, float *output, int len) {


  //*** General Thread Info ***//
  int bx = blockIdx.x;
  int tx = threadIdx.x;


  //*** Generate reductionSegment ***//
  __shared__ float reductionSegment[(BLOCK_SIZE << 1)];

  // Coalesced read from global memory
  for (int k = 0; k < 2; k++) {
    int inputIndex = tx + (bx*(BLOCK_SIZE<<1)) + (k*BLOCK_SIZE);
    float inputValue = inputIndex < len ? input[inputIndex] : 0;
    reductionSegment[tx + (k*BLOCK_SIZE)] = inputValue;
    __syncthreads();
  }


  //*** Perform reduction ***//
  int rSx = tx * 2;
  for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {

    // only perform reduction with necessary threads, fewer needed over time
    // each iteration uses half as many as the previous
    if (tx % stride == 0) {
      reductionSegment[rSx] += reductionSegment[rSx + stride];
    }
    __syncthreads();
  }


  //*** Write result to output ***//
  if (tx == 0) {
    output[bx] = reductionSegment[0];
  }
}



int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);



  //*** Importing data and creating memory on host ***//
  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);
  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");



  wbLog(TRACE, "The number of input elements in the input is ",  numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);



  //*** Allocating GPU memory ***//
  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void**) &deviceInput,  numInputElements  * sizeof(float));
  cudaMalloc((void**) &deviceOutput, numOutputElements * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");



  //*** Copying input memory to the GPU ***//
  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");



  //*** Performing CUDA computation ***//
  wbTime_start(Compute, "Performing CUDA computation");
  dim3 dimGrid(numOutputElements, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  total<<<dimGrid, dimBlock>>>(
    deviceInput, deviceOutput, numInputElements
  );
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");



  //*** Copying output memory to the CPU ***//
  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");


  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. For simplicity, we do not
   * require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }



  //*** Freeing GPU Memory ***//
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");



  //*** Check Solution ***//
  wbSolution(args, hostOutput, 1);



  //*** Freeing CPU Memory ***//
  free(hostInput);
  free(hostOutput);



  //*** Exit ***//
  return 0;
}

