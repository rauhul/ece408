
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <cudnn.h>
#define TILE_WIDTH 16


//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

//////////////////////////////////////////////////////////////////////////////



namespace mxnet
{
namespace op
{


template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x, const DType *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    (void)W_out; // silence declared but never referenced warning. remove this line when you start working
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    const int W_grid = ceil(W_out / (double) TILE_WIDTH);

    int n, m, h, w, c, p, q;
    n = blockIdx.x;
    m = blockIdx.y;
    h = (blockIdx.z / W_grid) * blockDim.y + threadIdx.y;
    w = (blockIdx.z % W_grid) * blockDim.x + threadIdx.x;

    if(h < H_out && w < W_out) {
        float acc = 0;

        for(c = 0; c < C; c++)
            for(p = 0; p < K; p++)
                for(q = 0; q < K; q++)
                    acc += x4d(n, c, h+p, w+q) * k4d(m, c, p, q);

        y4d(n,m,h,w) = acc;
    }

    #undef y4d
    #undef x4d
    #undef k4d
}

// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    cudnnHandle_t handle_;
    checkCUDNN(cudnnCreate(&handle_));

    checkCUDNN(cudnnSetStream(handle_,                                  // cudnnHandle_t
                              s));                                      // cudaStream_t

    cudnnTensorDescriptor_t srcDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&srcDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(srcDesc,                      // cudnnTensorDescriptor_t
                                          CUDNN_TENSOR_NCHW,            // cudnnTensorFormat_t
                                          CUDNN_DATA_FLOAT,             // cudnnDataType_t
                                          (int) x.shape_[0],            // examples
                                          (int) x.shape_[1],            // channels
                                          (int) x.shape_[2],            // data height
                                          (int) x.shape_[3]));          // data width


    cudnnTensorDescriptor_t destDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&destDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(destDesc,                     // cudnnTensorDescriptor_t
                                          CUDNN_TENSOR_NCHW,            // cudnnTensorFormat_t
                                          CUDNN_DATA_FLOAT,             // cudnnDataType_t
                                          (int) y.shape_[0],            // examples
                                          (int) y.shape_[1],            // channels
                                          (int) y.shape_[2],            // data height
                                          (int) y.shape_[3]));          // data width


    cudnnFilterDescriptor_t filterDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,                   // cudnnFilterDescriptor_t
                                          CUDNN_DATA_FLOAT,             // cudnnDataType_t
                                          CUDNN_TENSOR_NCHW,            // cudnnTensorFormat_t
                                          (int) w.shape_[0],            // k
                                          (int) w.shape_[1],            // c
                                          (int) w.shape_[2],            // filter height
                                          (int) w.shape_[3]));          // filter width


    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,                // cudnnConvolutionDescriptor_t
                                               0,                       // horizontal padding
                                               0,                       // vertical padding
                                               1,                       // horizontal stride
                                               1,                       // vertical stride
                                               1,                       // horizontal scaling
                                               1,                       // vertical scaling
                                               CUDNN_CONVOLUTION,       // cudnnConvolutionMode_t
                                               CUDNN_DATA_FLOAT));      // cudnnDataType_t


    size_t workSpaceSize;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle_,         // cudnnHandle_t
                                                       srcDesc,         // cudnnTensorDescriptor_t
                                                       filterDesc,      // cudnnFilterDescriptor_t
                                                       convDesc,        // cudnnConvolutionDescriptor_t
                                                       destDesc,        // cudnnTensorDescriptor_t
                                                       CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, // cudnnConvolutionFwdAlgo_t
                                                       &workSpaceSize));// workSpaceSizeInBytes

    std::cout << "workSpaceSize: " << workSpaceSize << std::endl;

    void *workSpace;
    checkCudaErrors(cudaMalloc(&workSpace, workSpaceSize));

    int alpha = 1;
    int beta = 0;
    checkCUDNN(cudnnConvolutionForward(handle_,                        // cudnnHandle_t
                                       &alpha,                         // alpha
                                       srcDesc,                        // cudnnTensorDescriptor_t
                                       x.dptr_,                        // srcData
                                       filterDesc,                     // cudnnFilterDescriptor_t,
                                       w.dptr_,                        // filterData
                                       convDesc,                       // cudnnConvolutionDescriptor_t
                                       CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, // cudnnConvolutionFwdAlgo_t
                                       workSpace,                      // workSpace
                                       workSpaceSize,                  // workSpaceSizeInBytes
                                       &beta,                          // beta
                                       destDesc,                       // cudnnTensorDescriptor_t
                                       y.dptr_));                      // destData

    checkCUDNN(cudnnDestroy(handle_));
    checkCudaErrors(cudaFree(workSpace));

    // MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



}
}

#endif
