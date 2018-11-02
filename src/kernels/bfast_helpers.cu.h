#ifndef BFAST_HELPERS_CU_H
#define BFAST_HELPERS_CU_H

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Matrix transposition (naive)

static __global__ void transpose_kernel_NAIVE(float *A, float *B,
                                              int heightA, int widthA)
{
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;

  if(gidx >= widthA || gidy >= heightA) { return; }

  B[IDX_2D(gidx, gidy, heightA)] = A[IDX_2D(gidy, gidx, widthA)];
}

static void transpose_NAIVE(void *d_A, void *d_B, int heightA, int widthA)
{
  dim3 block(16, 16, 1);
  dim3 grid(CEIL_DIV(widthA, block.x),
            CEIL_DIV(heightA, block.y),
            1);
  transpose_kernel_NAIVE<<<grid, block>>>((float *)d_A, (float *)d_B,
                                          heightA, widthA);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Matrix transposition (tiled)

template <class ElTp, int T>
__global__ void transpose_tiled_kernel(ElTp* A, ElTp* B,
                                       int heightA, int widthA)
{
  // Grid:  (CEIL_DIV(widthA, T), CEIL_DIV(heightA, T), 1)
  // Block: (T, T, 1)

  extern __shared__ char sh_mem1[];
  volatile ElTp *tile = (volatile ElTp *)sh_mem1;

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;

  if (x < widthA && y < heightA) {
    tile[IDX_2D(threadIdx.y, threadIdx.x, T + 1)] = A[IDX_2D(y, x, widthA)];
  }
  __syncthreads();

  x = blockIdx.y * T + threadIdx.x;
  y = blockIdx.x * T + threadIdx.y;
  if (x < heightA && y < widthA) {
    B[IDX_2D(y, x, heightA)] = tile[IDX_2D(threadIdx.x, threadIdx.y, T + 1)];
  }
}

template<class ElTp, int T>
void transpose_tiled(ElTp* d_in, ElTp* d_out,
                     const unsigned int height,
                     const unsigned int width)
{
   int dimy = CEIL_DIV(height, T);
   int dimx = CEIL_DIV(width, T);
   dim3 block(T, T, 1);
   dim3 grid (dimx, dimy, 1);
   unsigned int sh_mem_size = T * (T + 1) * sizeof(ElTp);

   transpose_tiled_kernel<ElTp,T><<<grid, block, sh_mem_size>>>
                                 (d_in, d_out, height, width);
}

static void transpose(void *d_A, void *d_B, int heightA, int widthA)
{
  transpose_tiled<float, 32>((float *)d_A, (float *)d_B, heightA, widthA);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Inclusive scan with (+)

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
__device__ inline T scaninc_block_add_nowrite(volatile T *in)
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

  return val;
}

template <class T>
__device__ inline void scaninc_block_add(volatile T *in)
{
  const unsigned int idx = threadIdx.x;
  T val = scaninc_block_add_nowrite(in);
  __syncthreads();
  in[idx] = val;
  __syncthreads();
}


#endif
