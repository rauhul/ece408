
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <iomanip>

#define TILE_WIDTH 28

#define N_K 50
#define K   5

namespace mxnet {
namespace op {

__constant__ float kernels[N_K * K * K];


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
            X_unroll[w_unroll*576 + tx] = X[(h_out + p)*28 + (w_out + q)];//tx + q + (28*p)
        }
    }
}

template<typename gpu, typename DType>
__global__ void matmul_kernel(const DType* X, DType* Y) {

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int y  = threadIdx.y + 25*blockIdx.y;

    __shared__ DType tile[25][25];
    if (y < 576) {
        tile[tx][ty] = X[tx*576 + y];
    }
    __syncthreads();

    if (y < 576) {
        DType acc;
        acc = 0;
        #pragma unroll
        for (int i = 0; i < 25; ++i) {
            acc += tile[i][ty] * kernels[tx*25 + i];
        }
        Y[tx*576 + y] = acc;

        acc = 0;
        #pragma unroll
        for (int i = 0; i < 25; ++i) {
            acc += tile[i][ty] * kernels[(tx+25)*25 + i];
        }
        Y[(tx+25)*576 + y] = acc;
    }
}


template<typename gpu, typename DType>
__global__ void print_kernel(DType* X) {
    printf("%i: %f\n", threadIdx.x, X[threadIdx.x]);
}

template<typename gpu, typename DType>
__global__ void simmul_kernel(DType* X, DType* Y) {
    int col = blockIdx.x;
    int row = blockIdx.y;

    float acc = 0;
    for (int i = 0; i < 25; ++i) {
        acc += X[row + i*576] * kernels[col*25 + i];
    }

    Y[576*col + row] = acc;
}


// Called by new-inl.h
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    cudaStream_t s = y.stream_->stream_;

    const int B = x.shape_[0];

    cudaMemcpyToSymbol(kernels, w.dptr_, N_K * K * K * sizeof(float), 0, cudaMemcpyHostToDevice);

    dim3 unroll_grid(1, 1, 1);
    dim3 unroll_block(576, 1, 1);

    dim3 matmul_grid(1, 24, 1);
    dim3 matmul_block(25, 25, 1);

    dim3 simmul_grid(50, 576, 1);
    dim3 simmul_block(1, 1, 1);



    DType *x_unrolled;
    cudaMalloc((void**) &x_unrolled, 25 * 576 * sizeof(DType));

    for (int n = 0; n < B; ++n) {
        unroll_kernel<gpu, DType><<<unroll_grid, unroll_block, 0, s>>>(x.dptr_ + (n * 784), x_unrolled);
        cudaDeviceSynchronize();

        // simmul_kernel<gpu, DType><<<simmul_grid, simmul_block, 0, s>>>(x_unrolled, y.dptr_ + (n * 576 * 50));

        matmul_kernel<gpu, DType><<<matmul_grid, matmul_block, 0, s>>>(x_unrolled, y.dptr_ + (n * 576 * 50));
        cudaDeviceSynchronize();
    }


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
