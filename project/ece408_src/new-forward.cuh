
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <iomanip>

#define TILE_WIDTH 28

#define N_K 50
#define K   5

namespace mxnet {
namespace op {

// __constant__ float kernels[N_K * K * K];

#define UNROLL_GROUP_SIZE 2000

template<typename gpu, typename DType>
__global__ void unroll_kernel(const DType* X, DType* X_unroll) {
    int w_unroll, p, q;

    int tx = threadIdx.x;

    int h_out = tx / 24;
    int w_out = tx % 24;

    #pragma unroll
    for (p = 0; p < K; ++p) {
        #pragma unroll
        for (q = 0; q < K; ++q) {//(h_out + p)*28 + (w_out + q)
            w_unroll = p * K + q;
            X_unroll[(blockIdx.x * 576 * 25) + w_unroll*576 + tx] = X[(blockIdx.x * 784) + (h_out + p)*28 + (w_out + q)];//tx + q + (28*p)
        }
    }
}

template<typename gpu, typename DType>
__global__ void matmul_kernel(const DType* X, DType* Y, DType* W) {

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int y  = threadIdx.y + 25*blockIdx.y;

    __shared__ DType tile[25][25];
    __shared__ DType filter[N_K * K * K];

    if (y < 576) {
        tile[tx][ty] = X[(blockIdx.x * 576 * 25) + tx*576 + y];
    }

    __syncthreads();
    filter[(ty+00)*25 + tx] = W[(ty+00)*25 + tx];
    filter[(ty+25)*25 + tx] = W[(ty+25)*25 + tx];
    __syncthreads();

    if (y < 576) {
        DType acc;
        acc = 0;
        #pragma unroll
        for (int i = 0; i < 25; ++i) {
            acc += tile[i][ty] * filter[tx*25 + i];
        }
        Y[(blockIdx.x * 576 * 50) + tx*576 + y] = acc;

        acc = 0;
        #pragma unroll
        for (int i = 0; i < 25; ++i) {
            acc += tile[i][ty] * filter[(tx+25)*25 + i];
        }
        Y[(blockIdx.x * 576 * 50) + (tx+25)*576 + y] = acc;
    }
}

// Called by new-inl.h
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    // cudaStream_t s = y.stream_->stream_;

    const int B = x.shape_[0];

    dim3 unroll_grid(UNROLL_GROUP_SIZE, 1, 1);
    dim3 unroll_block(576, 1, 1);

    dim3 matmul_grid(UNROLL_GROUP_SIZE, 24, 1);
    dim3 matmul_block(25, 25, 1);


    DType *x_unrolled;
    cudaMalloc((void**) &x_unrolled, 10000 * 25 * 576 * sizeof(DType));

    cudaStream_t stream0;
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStream_t stream3;
    cudaStream_t stream4;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    unroll_kernel<gpu, DType><<<unroll_grid, unroll_block, 0, stream0>>>(x.dptr_    + (0 * UNROLL_GROUP_SIZE * 784),        x_unrolled + (0 * UNROLL_GROUP_SIZE * 576 * 25));
    matmul_kernel<gpu, DType><<<matmul_grid, matmul_block, 0, stream0>>>(x_unrolled + (0 * UNROLL_GROUP_SIZE * 576 * 25),   y.dptr_    + (0 * UNROLL_GROUP_SIZE * 576 * 50), w.dptr_);

    unroll_kernel<gpu, DType><<<unroll_grid, unroll_block, 0, stream1>>>(x.dptr_    + (1 * UNROLL_GROUP_SIZE * 784),        x_unrolled + (1 * UNROLL_GROUP_SIZE * 576 * 25));
    matmul_kernel<gpu, DType><<<matmul_grid, matmul_block, 0, stream1>>>(x_unrolled + (1 * UNROLL_GROUP_SIZE * 576 * 25),   y.dptr_    + (1 * UNROLL_GROUP_SIZE * 576 * 50), w.dptr_);

    unroll_kernel<gpu, DType><<<unroll_grid, unroll_block, 0, stream2>>>(x.dptr_    + (2 * UNROLL_GROUP_SIZE * 784),        x_unrolled + (2 * UNROLL_GROUP_SIZE * 576 * 25));
    matmul_kernel<gpu, DType><<<matmul_grid, matmul_block, 0, stream2>>>(x_unrolled + (2 * UNROLL_GROUP_SIZE * 576 * 25),   y.dptr_    + (2 * UNROLL_GROUP_SIZE * 576 * 50), w.dptr_);

    unroll_kernel<gpu, DType><<<unroll_grid, unroll_block, 0, stream3>>>(x.dptr_    + (3 * UNROLL_GROUP_SIZE * 784),        x_unrolled + (3 * UNROLL_GROUP_SIZE * 576 * 25));
    matmul_kernel<gpu, DType><<<matmul_grid, matmul_block, 0, stream3>>>(x_unrolled + (3 * UNROLL_GROUP_SIZE * 576 * 25),   y.dptr_    + (3 * UNROLL_GROUP_SIZE * 576 * 50), w.dptr_);

    unroll_kernel<gpu, DType><<<unroll_grid, unroll_block, 0, stream4>>>(x.dptr_    + (4 * UNROLL_GROUP_SIZE * 784),        x_unrolled + (4 * UNROLL_GROUP_SIZE * 576 * 25));
    matmul_kernel<gpu, DType><<<matmul_grid, matmul_block, 0, stream4>>>(x_unrolled + (4 * UNROLL_GROUP_SIZE * 576 * 25),   y.dptr_    + (4 * UNROLL_GROUP_SIZE * 576 * 50), w.dptr_);

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    cudaStreamSynchronize(stream4);


    // dim3 grid(1, 1, 1);
    // dim3 block(784, 1, 1);

    // print_kernel<gpu, DType><<<grid, block, 0, s>>>(x.dptr_);
    // cudaDeviceSynchronize();



    // int n = 0;
    // unroll_kernel<gpu, DType><<<unroll_grid, unroll_block, 0 , s>>>(x.dptr_ + (n * 28 * 28), x_unrolled);



    // for (int i = 0; i < 576; ++i) {
    //     for (int k = 0; k < 25; ++k) {
    //         std::cout << data[i + k*576] << ", ";
    //     }
    //     std::cout << std::endl;
    // }

    // dim3 a(576, 1, 1);
    // dim3 b(1, 1, 1);

    // print_kernel<gpu, DType><<<b, a, 0, s>>>(x_unrolled);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    cudaFree(x_unrolled);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);

}



}
}
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
#endif
