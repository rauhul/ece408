
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 28

namespace mxnet
{
namespace op
{

// __constant__ float *kernels;

template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x, const DType *k, const int B, const int M, const int H, const int W, const int K) {

    const int H_out = 24;
    const int W_out = 24;

    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i1,i0)    x[(i3) * (H * W) + (i1)*(W) + i0]
    #define k4d(i3,i1,i0)    k[(i3) * (K * K) + (i1)*(K) + i0]

    int n = blockIdx.x;
    int m = blockIdx.y;
    int h = threadIdx.y;
    int w = threadIdx.x;


    __shared__ float tileCache[TILE_WIDTH][TILE_WIDTH];
    tileCache[h][w] = x4d(n, h, w);
    __syncthreads();

    if(h < H_out && w < W_out) {
        float acc = 0;

        for(int p = 0; p < K; ++p)
            for(int q = 0; q < K; ++q)
                acc += tileCache[h+p][w+q] * k4d(m, p, q);

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

//     std::cout << "x" << std::endl;
//     std::cout << "x.shape_[0]" << x.shape_[0] << std::endl;
//     std::cout << "x.shape_[1]" << x.shape_[1] << std::endl;
//     std::cout << "x.shape_[2]" << x.shape_[2] << std::endl;
//     std::cout << "x.shape_[3]" << x.shape_[3] << std::endl;

//     std::cout << "w" << std::endl;
//     std::cout << "w.shape_[0]" << w.shape_[0] << std::endl;
//     std::cout << "w.shape_[1]" << w.shape_[1] << std::endl;
//     std::cout << "w.shape_[2]" << w.shape_[2] << std::endl;
//     std::cout << "w.shape_[3]" << w.shape_[3] << std::endl;

//     std::cout << "y" << std::endl;
//     std::cout << "y.shape_[0]" << y.shape_[0] << std::endl;
//     std::cout << "y.shape_[1]" << y.shape_[1] << std::endl;
//     std::cout << "y.shape_[2]" << y.shape_[2] << std::endl;
//     std::cout << "y.shape_[3]" << y.shape_[3] << std::endl;

// x
// x.shape_[0]10000
// x.shape_[1]1
// x.shape_[2]28
// x.shape_[3]28
// w
// w.shape_[0]50
// w.shape_[1]1
// w.shape_[2]5
// w.shape_[3]5
// y
// y.shape_[0]10000
// y.shape_[1]50
// y.shape_[2]24
// y.shape_[3]24

    // Use mxnet's CHECK_EQ to do assertions.
    //CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...

    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, 1);

    // Call the kernel
    forward_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



}
}

#endif
