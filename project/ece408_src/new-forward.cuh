
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 16

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

    /*
        Your code here!
    */
	const int W_grid = ceil(W_out / (double) TILE_WIDTH);

	int n, m, h, w, c, p, q;
	n = blockIdx.x;
	m = blockIdx.y;
	h = (blockIdx.z / W_grid) * blockDim.y + threadIdx.y;
	w = (blockIdx.z % W_grid) * blockDim.x + threadIdx.x;
	//if(m == 0 && n == 0)
		//printf("hi rabool: %i %i\n", h, w);

	if(h < H_out && w < W_out)
	{
		float acc = 0;

		for(c = 0; c < C; c++)
		{
			for(p = 0; p < K; p++)
			{
				for(q = 0; q < K; q++)
				{
					acc += x4d(n, c, h+p, w+q) * k4d(m, c, p, q);
				}
			}
		}
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
	const int W_grid = ceil(W_out / (double) TILE_WIDTH);
	const int H_grid = ceil(H_out / (double) TILE_WIDTH);
	const int Z = H_grid * W_grid;
	
	printf("Values: %i,%i,%i,%i,%i\n", H_out, W_out, W_grid, H_grid, Z);

	dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 gridDim(B, M, Z);

    // Call the kernel
    forward_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



}
}

#endif
