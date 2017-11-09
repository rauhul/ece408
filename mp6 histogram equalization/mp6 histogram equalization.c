// Histogram Equalization
#include <wb.h>

//*** Typedefs ***//
typedef unsigned char uint8_t;
typedef unsigned int  uint_t;

//*** Macros ***//

//*** Program-wide constants ***//
#define NUM_CHANNELS 3
#define HISTOGRAM_LENGTH 256

//*** Device constant memory ***//

//*** CUDA Kernels ***//
//Cast the image to unsigned char
__global__ void floatToUInt8(float *input, uint8_t *output,
                              int width, int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = blockIdx.z * (width * height) + y * (width) + x;
    output[idx] = (uint8_t) ((HISTOGRAM_LENGTH - 1) * input[idx]);
  }
}

//Convert the image from RGB to Gray Scale.
__global__ void rgbToGrayScale(uint8_t *input, uint8_t *output,
                                int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = y * (width) + x;
    uint8_t r = input[3 * idx + 0];
    uint8_t g = input[3 * idx + 1];
    uint8_t b = input[3 * idx + 2];
    output[idx] = (uint8_t) (0.21*r + 0.71*g + 0.07*b);
  }
}

//Compute the histogram of the image
__global__ void grayScaleToHistogram(uint8_t *input, uint_t *output,
                                      int width, int height) {

  __shared__ uint_t histogram[HISTOGRAM_LENGTH];

  int tIdx = threadIdx.x + threadIdx.y * blockDim.x;
  if (tIdx < HISTOGRAM_LENGTH) {
    histogram[tIdx] = 0;
  }

  __syncthreads();
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y * (width) + x;
    uint8_t val = input[idx];
    atomicAdd(&(histogram[val]), 1);
  }

  __syncthreads();
  if (tIdx < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[tIdx]), histogram[tIdx]);
  }
}

//Compute the scan (prefix sum) of the histogram to arrive at the histogram equalization function
__global__ void histogramToCDF(uint_t *input, float *output,
                                int width, int height) {
  __shared__ uint_t cdf[HISTOGRAM_LENGTH];
  int x = threadIdx.x;
  cdf[x] = input[x];

  //** Parallel inclusive scan algorithm (based on Brent-Kung) **//
  // First scan half
  for (unsigned int stride = 1; stride <= HISTOGRAM_LENGTH / 2; stride *= 2) {
    __syncthreads();
    int idx = (x + 1) * 2 * stride - 1;
    if (idx < HISTOGRAM_LENGTH) {
      cdf[idx] += cdf[idx - stride];
    }
  }

  // Second scan half
  for (int stride = HISTOGRAM_LENGTH / 4; stride > 0; stride /= 2) {
    __syncthreads();
    int idx = (x + 1) * 2 * stride - 1;
    if (idx + stride < HISTOGRAM_LENGTH) {
      cdf[idx + stride] += cdf[idx];
    }
  }

  __syncthreads();
  output[x] = cdf[x] / ((float) (width * height));
}

//Apply the equalization function to the input image to get the color corrected image
__global__ void equalizeImage(uint8_t *inout, float *cdf,
                               int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = blockIdx.z * (width * height) + y * (width) + x;
    uint8_t val = inout[idx];

    float equalized = 255 * (cdf[val] - cdf[0]) / (1.0 - cdf[0]);
    float clamped   = min(max(equalized, 0.0), 255.0);

    inout[idx] = (uint8_t) (clamped);
  }
}

//Cast the image to float
__global__ void uInt8ToFloat(uint8_t *input, float *output,
                              int width, int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = blockIdx.z * (width * height) + y * (width) + x;
    output[idx] = (float) (input[idx] / 255.0);
  }
}


//*** CPU Code ***//
int main(int argc, char **argv) {

  wbArg_t args;
  const char *inputImageFile;

  int imageWidth;
  int imageHeight;
  int imageChannels;

  wbImage_t inputImage;
  wbImage_t outputImage;

  float *hostInputImageData;
  float *hostOutputImageData;

  float   *deviceImageFloat;
  uint8_t *deviceImageUChar;
  uint8_t *deviceImageUCharGrayScale;
  uint_t  *deviceImageHistogram;
  float   *deviceImageCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage          = wbImport(inputImageFile);
  imageWidth          = wbImage_getWidth(inputImage);
  imageHeight         = wbImage_getHeight(inputImage);
  imageChannels       = wbImage_getChannels(inputImage);
  hostInputImageData  = wbImage_getData(inputImage);
  outputImage         = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  printf("%d, %d, %d\n", imageWidth, imageHeight, imageChannels);

  assert(imageChannels == NUM_CHANNELS);

  //*** Allocating GPU memory ***//
  cudaMalloc((void**) &deviceImageFloat,          imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceImageUChar,          imageWidth * imageHeight * imageChannels * sizeof(uint8_t));
  cudaMalloc((void**) &deviceImageUCharGrayScale, imageWidth * imageHeight *                 sizeof(uint8_t));
  cudaMalloc((void**) &deviceImageHistogram,      HISTOGRAM_LENGTH *                         sizeof(uint_t));
  cudaMemset((void *) deviceImageHistogram, 0,    HISTOGRAM_LENGTH *                         sizeof(uint_t));
  cudaMalloc((void**) &deviceImageCDF,            HISTOGRAM_LENGTH *                         sizeof(float));


  //*** Copying input memory to the GPU ***//
  cudaMemcpy(deviceImageFloat, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  //*** Performing CUDA computation ***//
  dim3 dimGrid;
  dim3 dimBlock;


  //to uint8
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32, 32, 1);

  floatToUInt8<<<dimGrid, dimBlock>>>(
    deviceImageFloat, deviceImageUChar,
    imageWidth, imageHeight
  );
  cudaDeviceSynchronize();


  //to gray
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dimBlock = dim3(32, 32, 1);

  rgbToGrayScale<<<dimGrid, dimBlock>>>(
    deviceImageUChar, deviceImageUCharGrayScale,
    imageWidth, imageHeight
  );
  cudaDeviceSynchronize();


  //to histo
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dimBlock = dim3(32, 32, 1);

  grayScaleToHistogram<<<dimGrid, dimBlock>>>(
    deviceImageUCharGrayScale, deviceImageHistogram,
    imageWidth, imageHeight
  );
  cudaDeviceSynchronize();


  //to cdf
  dimGrid  = dim3(1, 1, 1);
  dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);

  histogramToCDF<<<dimGrid, dimBlock>>>(
    deviceImageHistogram, deviceImageCDF,
    imageWidth, imageHeight
  );
  cudaDeviceSynchronize();


  //equalize
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32, 32, 1);

  equalizeImage<<<dimGrid, dimBlock>>>(
    deviceImageUChar, deviceImageCDF,
    imageWidth, imageHeight
  );
  cudaDeviceSynchronize();


  //to uint8
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32, 32, 1);

  uInt8ToFloat<<<dimGrid, dimBlock>>>(
    deviceImageUChar, deviceImageFloat,
    imageWidth, imageHeight
  );
  cudaDeviceSynchronize();


  //*** Copying output memory to the CPU ***//
  cudaMemcpy(hostOutputImageData, deviceImageFloat,
             imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  //*** Check Solution ***//
  wbSolution(args, outputImage);

  //*** Freeing GPU Memory ***//
  cudaFree(deviceImageFloat);
  //cudaFree(deviceOutputImageData);

  //*** Freeing CPU Memory ***//

  //*** Exit ***//
  return 0;
}

