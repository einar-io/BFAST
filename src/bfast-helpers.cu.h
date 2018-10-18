#include <cstdlib>

#define CUDA_SUCCEED(x) cuda_api_succeed(x, #x, __FILE__, __LINE__)

static inline void cuda_api_succeed(cudaError res, const char *call,
    const char *file, int line)
{
  if (res != cudaSuccess) {
    fprintf(stderr, "%s:%d: CUDA CALL\n  %s\nfailed with error code %d (%s)\n",
        file, line, call, res, cudaGetErrorString(res));
    exit(EXIT_FAILURE);
  }
}

#define IDX_2D(__r,__c,__nc) ((__r) * (__nc) + (__c))
#define CEIL_DIV(a,b) (((a) + (b) - 1) / (b))

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Matrix transposition

__global__ void transpose_kernel(float *A, float *B, int heightA, int widthA)
{

  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;

  if(gidx >= widthA || gidy >= heightA) {
    return;
  }

  B[IDX_2D(gidx, gidy, heightA)] = A[IDX_2D(gidy, gidx, widthA)];
}

void transpose(float *d_A, float *d_B, int heightA, int widthA)
{
  dim3 block(16, 16, 1);
  dim3 grid(CEIL_DIV(widthA, block.x),
            CEIL_DIV(heightA, block.y),
            1);
  transpose_kernel<<<grid, block>>>(d_A, d_B, heightA, widthA);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// For bfast_5 and bfast_6

template <class T>
__device__ inline T scaninc_warp_add(volatile T *in)
{
  const unsigned int idx  = threadIdx.x;
  const unsigned int lane = idx & 31;

  // no synchronization needed inside a WARP,
  //   i.e., SIMD execution
  if (lane >= 1)  { in[idx] = in[idx-1]  + in[idx]; }
  if (lane >= 2)  { in[idx] = in[idx-2]  + in[idx]; }
  if (lane >= 4)  { in[idx] = in[idx-4]  + in[idx]; }
  if (lane >= 8)  { in[idx] = in[idx-8]  + in[idx]; }
  if (lane >= 16) { in[idx] = in[idx-16] + in[idx]; }

  return in[idx];
}

template <class T>
__device__ inline void scaninc_block_add(volatile T *in)
{
  const unsigned int idx    = threadIdx.x;
  const unsigned int lane   = idx &  31;
  const unsigned int warpid = idx >> 5;

  T val = scaninc_warp_add(in);
  __syncthreads();

  if (lane == 31) { in[warpid] = val; }
  __syncthreads();

  if (warpid == 0) scaninc_warp_add(in);
  __syncthreads();

  if (warpid > 0) {
    val = in[warpid-1] + val;
  }

  __syncthreads();
  in[idx] = val;
  __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// For bfast_6 and bfast_7a

/*

float op2(float a, float b) { return a + b; }

float scaninc_warp_op2(volatile float *in, volatile float *out)
{
  const unsigned int idx = threadIdx.x;
  const unsigned int lane = idx & 31;

  // no synchronization needed inside a WARP,
  //   i.e., SIMD execution
  if (lane >= 1)  out[idx] = op2(in[idx-1],  in[idx]);
  if (lane >= 2)  out[idx] = op2(in[idx-2],  in[idx]);
  if (lane >= 4)  out[idx] = op2(in[idx-4],  in[idx]);
  if (lane >= 8)  out[idx] = op2(in[idx-8],  in[idx]);
  if (lane >= 16) out[idx] = op2(in[idx-16], in[idx]);

  return out[idx];
}

void scaninc_block_op2(volatile float *in, volatile float *out)
{
  const unsigned int idx = threadIdx.x
  const unsigned int lane = idx &  31;
  const unsigned int warpid = idx >> 5;

  float val = scaninc_map_warp(in, out);
  __syncthreads();

  if (lane == 31) { out[warpid] = val; }
  __syncthreads();

  if (warpid == 0) { scaninc_warp(out); }
  __syncthreads();

  if (warpid > 0) {
      val = op(out[warpid-1], val);
  }

  __syncthreads();
  out[idx] = val;
  __syncthreads();
}
*/
