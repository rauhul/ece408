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

// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

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


    cudnnConvolutionFwdAlgo_t algo;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(handle_,             // cudnnHandle_t
                                                   srcDesc,             // cudnnTensorDescriptor_t
                                                   filterDesc,          // cudnnFilterDescriptor_t
                                                   convDesc,            // cudnnConvolutionDescriptor_t
                                                   destDesc,            // cudnnTensorDescriptor_t
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, // cudnnConvolutionFwdPreference_t
                                                   0,                   // memoryLimitInBytes
                                                   &algo));             // cudnnConvolutionFwdAlgo_t


    size_t workSpaceSize;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle_,         // cudnnHandle_t
                                                       srcDesc,         // cudnnTensorDescriptor_t
                                                       filterDesc,      // cudnnFilterDescriptor_t
                                                       convDesc,        // cudnnConvolutionDescriptor_t
                                                       destDesc,        // cudnnTensorDescriptor_t
                                                       algo,            // cudnnConvolutionFwdAlgo_t
                                                       &workSpaceSize));// workSpaceSizeInBytes

    std::cout << "workSpaceSize: " << workSpaceSize << std::endl;

    void *workSpace;
    checkCudaErrors(cudaMalloc(&workSpace, workSpaceSize));

    int alpha = 1;
    int beta = 0;
    checkCUDNN(cudnnConvolutionForward(handle_,                         // cudnnHandle_t
                                       &alpha,                          // alpha
                                       srcDesc,                         // cudnnTensorDescriptor_t
                                       x.dptr_,                         // srcData
                                       filterDesc,                      // cudnnFilterDescriptor_t,
                                       w.dptr_,                         // filterData
                                       convDesc,                        // cudnnConvolutionDescriptor_t
                                       algo,                            // cudnnConvolutionFwdAlgo_t
                                       workSpace,                       // workSpace
                                       workSpaceSize,                   // workSpaceSizeInBytes
                                       &beta,                           // beta
                                       destDesc,                        // cudnnTensorDescriptor_t
                                       y.dptr_));                       // destData

    // checkCUDNN(cudnnDestroy(handle_));
    checkCudaErrors(cudaFree(workSpace));

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



}
}

#endif
