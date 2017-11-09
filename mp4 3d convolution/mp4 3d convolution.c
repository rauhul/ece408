#include <wb.h>

//*** Macros ***//
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define inBounds(x, y, z) \
  ((0 <= (x) && (x) < x_size) && \
   (0 <= (y) && (y) < y_size) && \
   (0 <= (z) && (z) < z_size))

//*** Program-wide constants ***//
#define KERNEL_SIZE   3
#define KERNEL_RADIUS 1

#define TILE_SIZE     KERNEL_SIZE
#define CACHE_SIZE    (KERNEL_SIZE + (KERNEL_RADIUS * 2))

//*** Device constant memory ***//
__constant__ float deviceKernel[KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE];

//*** 3D convolution kernel ***//
__global__ void conv3d(float *input, float *output,
                        const int z_size, const int y_size, const int x_size) {

  // General Thread Info
  int bx = blockIdx.x * TILE_SIZE; int tx = threadIdx.x;
  int by = blockIdx.y * TILE_SIZE; int ty = threadIdx.y;
  int bz = blockIdx.z * TILE_SIZE; int tz = threadIdx.z;

  //*** Generate tileCache ***//
  __shared__ float tileCache[CACHE_SIZE][CACHE_SIZE][CACHE_SIZE];

  // map each thread to a position in the kernel
  int tid = tz * (KERNEL_SIZE * KERNEL_SIZE) + ty * (KERNEL_SIZE) + tx;
  if (tid < CACHE_SIZE * CACHE_SIZE) {

    // map each kernel position to location in tile cache
    int tileX =  tid % CACHE_SIZE;
    int tileY = (tid / CACHE_SIZE) % CACHE_SIZE;

    int inputX = bx + tileX - 1;
    int inputY = by + tileY - 1;
    int inputZPartial = bz - 1;
    int inputZ;

    // load part of the tile cache
    for (int i = 0; i < CACHE_SIZE; i += 1) {
      inputZ = inputZPartial + i;

      if (inBounds(inputX, inputY, inputZ)) {
        tileCache[tileX][tileY][i] = input[inputZ * (y_size * x_size) + inputY * (x_size) + inputX];
      } else {
        tileCache[tileX][tileY][i] = 0;
      }
    }
  }

  __syncthreads();

  //*** Perform block convolution ***//
  // Exit threads outside of matrix boundry
  int xPos = bx + tx;
  int yPos = by + ty;
  int zPos = bz + tz;

  if (inBounds(xPos, yPos, zPos)) {
    float outputValue = 0;
    for (int x = 0; x < KERNEL_SIZE; x += 1) {
      for (int y = 0; y < KERNEL_SIZE; y += 1) {
        for (int z = 0; z < KERNEL_SIZE; z += 1) {
            outputValue +=
              tileCache[tx + x][ty + y][tz + z] *
              deviceKernel[z * (KERNEL_SIZE * KERNEL_SIZE) + y * (KERNEL_SIZE) + x];
        }
      }
    }
    output[zPos * (y_size * x_size) + yPos * (x_size) + xPos] = outputValue;
  }
}



int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;
  args = wbArg_read(argc, argv);



  // Import data
  hostInput  = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));



  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);



  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
    //*** Allocating GPU memory ***//
    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void**) &deviceInput,  z_size * y_size * x_size * sizeof(float));
    cudaMalloc((void**) &deviceOutput, z_size * y_size * x_size * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");



    //*** Copying input memory to the GPU ***//
    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInput, hostInput + 3,  z_size * y_size * x_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength * sizeof(float), 0, cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");



    //*** Performing CUDA computation ***//
    wbTime_start(Compute, "Doing the computation on the GPU");
    dim3 dimGrid(ceil(x_size/double(TILE_SIZE)), ceil(y_size/double(TILE_SIZE)), ceil(z_size/double(TILE_SIZE)));
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    conv3d<<<dimGrid, dimBlock>>>(
      deviceInput, deviceOutput,
      z_size, y_size, x_size
    );
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Doing the computation on the GPU");



    //*** Copying output memory to the CPU ***//
    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutput + 3, deviceOutput, z_size * y_size * x_size * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");
  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");



  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;



  //*** Check Solution ***//
  wbSolution(args, hostOutput, inputLength);



  //*** Freeing GPU Memory ***//
  cudaFree(deviceInput);
  cudaFree(deviceOutput);



  //*** Freeing CPU Memory ***//
  free(hostInput);
  free(hostOutput);



  //*** Exit ***//
  return 0;
}

