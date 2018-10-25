#include <cstdio>
#include <cstdint>
#include <cassert>
#include "bfast-helpers.cu.h"
//#define INVALID_INDEX (-1)

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Matrix transposition

template <class ElTp, int T>
__global__ void transpose_tiled_kernel(ElTp* A, ElTp* B,
                                       int heightA, int widthA)
{
  extern __shared__ char sh_mem1[];
  volatile ElTp *tile = (volatile ElTp *)sh_mem1;

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;

  if (x < widthA && y < heightA) {
    tile[threadIdx.y*(T+1) + threadIdx.x] = A[y*widthA + x];
  }

  __syncthreads();

  x = blockIdx.y * T + threadIdx.x;
  y = blockIdx.x * T + threadIdx.y;

  if (x < heightA && y < widthA) {
    B[y*heightA + x] = tile[threadIdx.x*(T+1) + threadIdx.y];
  }
}

template<class ElTp, int T>
void transpose_tiled ( ElTp*              d_in,
                       ElTp*              d_out,
                       const unsigned int height,
                       const unsigned int width)
{
   // 1. setup block and grid parameters
   unsigned int sh_mem_size = T * (T+1) * sizeof(ElTp);
   int  dimy = (height+T-1) / T;
   int  dimx = (width +T-1) / T;
   dim3 block(T, T, 1);
   dim3 grid (dimx, dimy, 1);

   //2. execute the kernel
   transpose_tiled_kernel<ElTp,T><<<grid, block, sh_mem_size>>>
                                 (d_in, d_out, height, width);
   cudaDeviceSynchronize();
}

void transpose(float *d_A, float *d_B, int heightA, int widthA)
{
  transpose_tiled<float, 32>(d_A, d_B, heightA, widthA);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// For bfast_5, bfast_6, bfast_7a

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
//  Step 1: Generating X
//
// Output:
//   X: [k2p2][N]f32

__global__ void bfast_step_1(float *X, int k2p2, int N, float f)
{
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (gidy >= k2p2 || gidx >= N) {
    return;
  }

  int i = gidy;
  int j = gidx + 1;
  float val;
  if (i == 0) { val = 1.0; }
  else if (i == 1) { val = (float)j; }
  else {
    float angle = 2.0 * M_PI * (float)(i / 2) * (float)j / f;
    if (i % 2 == 0) {
      val = __sinf(angle);
    } else {
      val = __cosf(angle);
    }
  }

  X[IDX_2D(gidy, gidx, N)] = val;
}

extern "C" void bfast_step_1_single(float **X, int k2p2, int N, float f)
{
  float *d_X;
  const size_t mem_X = k2p2 * N * sizeof(float);

  CUDA_SUCCEED(cudaMalloc(&d_X, mem_X));
  dim3 block(16, 16, 1);
  dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(k2p2, block.y), 1);
  bfast_step_1<<<grid, block>>>(d_X, k2p2, N, f);

  *X = (float *)malloc(mem_X);
  CUDA_SUCCEED(cudaMemcpy(*X, d_X, mem_X, cudaMemcpyDeviceToHost));
  CUDA_SUCCEED(cudaFree(d_X));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Step 2: Calculating Xsqr
//
// Input:
//   X:    [k2p2][N]f32    (only using slice: [k2p2][n])
//   Xt:   [N][k2p2]f32    (only using slice: [n][k2p2])
//   Y:    [m][N]f32       (only using slice: [m][n])
// Output:
//   Xsqr: [m][k2p2][k2p2]f32
//

#define STEP_2_TILE_SIZE 28

__global__ void bfast_step_2(float *Xh, float *Xth, float *Yth, float *Xsqr,
    int N, int n, int k2p2, int m)
{
  // Grid: (CEIL_DIV(m, STEP_2_TILE_SIZE), 1, 1)
  // Block: (k2p2, k2p2, 1)

  if (threadIdx.y >= k2p2 || threadIdx.x >= k2p2) {
    return;
  }

  float accum[STEP_2_TILE_SIZE];
  __shared__ float ysh[STEP_2_TILE_SIZE];

  for (int t = 0; t < STEP_2_TILE_SIZE; t++) {
    accum[t] = 0.0;
  }

  for (int i = 0; i < n; i++) {
    float val = Xh[IDX_2D(threadIdx.y, i, N)]
                  * Xth[IDX_2D(i, threadIdx.x, k2p2)];

    int ysh_idx = IDX_2D(threadIdx.y, threadIdx.x, k2p2);
    if (ysh_idx < STEP_2_TILE_SIZE) {
      int y_row = blockIdx.x * STEP_2_TILE_SIZE + ysh_idx;
      if (y_row < m) {
        ysh[ysh_idx] = Yth[IDX_2D(i, y_row, N)];
      } else {
        ysh[ysh_idx] = 0.0;
      }
    }
    __syncthreads();

    for (int t = 0; t < STEP_2_TILE_SIZE; t++) {
      if (!isnan(ysh[t])) {
        accum[t] += val;
      }
    }
  }

  for (int t = 0; t < STEP_2_TILE_SIZE; t++) {
    int mat_idx = blockIdx.x * STEP_2_TILE_SIZE + t;
    if (mat_idx < m) {
      Xsqr[mat_idx * k2p2 * k2p2 + IDX_2D(threadIdx.y, threadIdx.x, k2p2)]
        = accum[t];
    }
  }
}

extern "C" void bfast_step_2_single(float *X, float *Xt, float *Y,
    float **Xsqr, int N, int n, int k2p2, int m)
{
  // XXX: This function should actually take Yt as input, not Y!
  float *d_X = NULL, *d_Xt = NULL, *d_Y = NULL, *d_Yt = NULL, *d_Xsqr = NULL;
  const size_t mem_X = k2p2 * N * sizeof(float);
  const size_t mem_Y = m * N * sizeof(float);
  const size_t mem_Xsqr = m * k2p2 * k2p2 * sizeof(float);

  CUDA_SUCCEED(cudaMalloc(&d_X, mem_X));
  CUDA_SUCCEED(cudaMalloc(&d_Xt, mem_X));
  CUDA_SUCCEED(cudaMalloc(&d_Y, mem_Y));
  CUDA_SUCCEED(cudaMalloc(&d_Yt, mem_Y));
  CUDA_SUCCEED(cudaMalloc(&d_Xsqr, mem_Xsqr));

  CUDA_SUCCEED(cudaMemcpy(d_X, X, mem_X, cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_Xt, Xt, mem_X, cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_Y, Y, mem_Y, cudaMemcpyHostToDevice));

  transpose(d_Y, d_Yt, m, N);

  dim3 block(8, 8, 1); // Assumes k2p2 <= 8
  dim3 grid(CEIL_DIV(m, STEP_2_TILE_SIZE), 1, 1);
  bfast_step_2<<<grid, block>>>(d_X, d_Xt, d_Yt, d_Xsqr, N, n, k2p2, m);

  *Xsqr = (float *)malloc(mem_Xsqr);
  CUDA_SUCCEED(cudaMemcpy(*Xsqr, d_Xsqr, mem_Xsqr, cudaMemcpyDeviceToHost));

  CUDA_SUCCEED(cudaFree(d_X));
  CUDA_SUCCEED(cudaFree(d_Xt));
  CUDA_SUCCEED(cudaFree(d_Y));
  CUDA_SUCCEED(cudaFree(d_Xsqr));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Step 3: Calculating Xinv
//
// Input:
//   Xsqr: [m][k2p2][k2p2]f32
// Output:
//   Xinv: [m][k2p2][k2p2]f32
//

__global__ void bfast_step_3(float *Xsqr, float *Xinv, int k2p2)
{
  // Grid: (m, 1, 1)
  // Block: (2*k2p2, k2p2, 1)
  // NB! Uses dynamically allocated shared memory: k2p2*2*k2p2 floats per block

  if (threadIdx.x >= 2*k2p2 || threadIdx.y >= k2p2) {
    return;
  }

  float *sqr = &Xsqr[blockIdx.x * (k2p2 * k2p2)];
  float *inv = &Xinv[blockIdx.x * (k2p2 * k2p2)];

  extern __shared__ float A[]; // [k2p2][2*k2p2]

  // Body of mat_inv map
  if (threadIdx.x < k2p2) {
    // Left half of A
    A[IDX_2D(threadIdx.y, threadIdx.x, 2*k2p2)] =
      sqr[IDX_2D(threadIdx.y, threadIdx.x, k2p2)];
  } else {
    // Right half of A
    float val = threadIdx.y == (threadIdx.x - k2p2) ? 1.0 : 0.0;
    A[IDX_2D(threadIdx.y, threadIdx.x, 2*k2p2)] = val;
  }
  __syncthreads();

  // guass_jordan loop and map body
  for (int i = 0; i < k2p2; i++) {
    float v1 = A[i];
    float x = A[threadIdx.x] / v1;
    float val = x;

    if (threadIdx.y < k2p2 - 1) {
      val = A[IDX_2D(threadIdx.y + 1, threadIdx.x, 2*k2p2)]
              - A[IDX_2D(threadIdx.y + 1, i, 2*k2p2)] * x;
    }
    __syncthreads();
    A[IDX_2D(threadIdx.y, threadIdx.x, 2*k2p2)] = val;
    __syncthreads();
  }

  // Write back to global memory
  if (threadIdx.x < k2p2) {
    inv[IDX_2D(threadIdx.y, threadIdx.x, k2p2)] =
      A[IDX_2D(threadIdx.y, threadIdx.x + k2p2, 2*k2p2)];
  }
}

extern "C" void bfast_step_3_single(float *Xsqr, float **Xinv, int k2p2, int m)
{
  float *d_Xsqr = NULL, *d_Xinv = NULL;
  const size_t mem_Xsqr = m * k2p2 * k2p2 * sizeof(float);

  CUDA_SUCCEED(cudaMalloc(&d_Xsqr, mem_Xsqr));
  CUDA_SUCCEED(cudaMalloc(&d_Xinv, mem_Xsqr));

  CUDA_SUCCEED(cudaMemcpy(d_Xsqr, Xsqr, mem_Xsqr, cudaMemcpyHostToDevice));

  dim3 block(16, 8, 1); // Assumes k2p2 <= 8
  dim3 grid(m, 1, 1);
  const size_t shared_size = k2p2 * 2 * k2p2 * sizeof(float);
  bfast_step_3<<<grid, block, shared_size>>>(d_Xsqr, d_Xinv, k2p2);

  *Xinv = (float *)malloc(mem_Xsqr);
  CUDA_SUCCEED(cudaMemcpy(*Xinv, d_Xinv, mem_Xsqr, cudaMemcpyDeviceToHost));
  CUDA_SUCCEED(cudaFree(d_Xinv));
  CUDA_SUCCEED(cudaFree(d_Xsqr));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Step 4a: Calculating beta0
//
// Input:
//   X:     [k2p2][N]f32    (only using slice: [k2p2][n])
//   Y:     [m][N]f32       (only using slice: [m][n])
// Output:
//   beta0: [m][k2p2]
//
// This calculation is performed by transposing Y (so its dimensions become
// [N][m]) and then applying (filtered) matrix-matrix multiplication.
// The output will need to be transposed again, since:
//      [k2p2][N] multiplied with [N][m] is [k2p2][m]
//

__global__ void bfast_step_4a(float *Xh, float *Yth, float *beta0t, int k2p2,
    int n, int m, int N)
{
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;

  if(gidy >= k2p2 || gidx >= m) {
    return;
  }

  float accum = 0.0;
  for(int k = 0; k < n; k ++) {
    float val = Yth[IDX_2D(k, gidx, m)];
    if (!isnan(val)) {
      accum += Xh[IDX_2D(gidy, k, N)] * val;
    }
  }

  beta0t[IDX_2D(gidy, gidx, m)] = accum;
}

extern "C" void bfast_step_4a_single(float *X, float *Y, float **beta0,
    int k2p2, int n, int m, int N)
{
  float *d_X = NULL, *d_Y = NULL, *d_Yt = NULL;
  float *d_beta0 = NULL, *d_beta0t = NULL;
  const size_t mem_X = k2p2 * N * sizeof(float);
  const size_t mem_Y = m * N * sizeof(float);
  const size_t mem_beta0 = m * k2p2 * sizeof(float);

  CUDA_SUCCEED(cudaMalloc(&d_X, mem_X));
  CUDA_SUCCEED(cudaMalloc(&d_Y, mem_Y));
  CUDA_SUCCEED(cudaMalloc(&d_Yt, mem_Y));
  CUDA_SUCCEED(cudaMalloc(&d_beta0, mem_beta0));
  CUDA_SUCCEED(cudaMalloc(&d_beta0t, mem_beta0));

  CUDA_SUCCEED(cudaMemcpy(d_X, X, mem_X, cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_Y, Y, mem_Y, cudaMemcpyHostToDevice));

  transpose(d_Y, d_Yt, m, N);

  dim3 block(16, 16, 1);
  dim3 grid(CEIL_DIV(m, block.x), CEIL_DIV(k2p2, block.y), 1);
  bfast_step_4a<<<grid, block>>>(d_X, d_Yt, d_beta0t, k2p2, n, m, N);

  transpose(d_beta0t, d_beta0, k2p2, m);

  *beta0 = (float *)malloc(mem_beta0);
  CUDA_SUCCEED(cudaMemcpy(*beta0, d_beta0, mem_beta0, cudaMemcpyDeviceToHost));
  CUDA_SUCCEED(cudaFree(d_X));
  CUDA_SUCCEED(cudaFree(d_Y));
  CUDA_SUCCEED(cudaFree(d_Yt));
  CUDA_SUCCEED(cudaFree(d_beta0));
  CUDA_SUCCEED(cudaFree(d_beta0t));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Step 4b: Calculating beta
//
// Input:
//   Xinv:  [m][k2p2][k2p2]f32
//   beta0: [m][k2p2]f32
// Output:
//   beta:  [m][k2p2]f32
//
// Every block performs a matrix-vector multiplication between a matrix from
// Xinv and a row from beta0. The resulting vectors are rows in the final
// [m][k2p2] matrix, beta.
//

__global__ void bfast_step_4b(float *Xinv, float *beta0, float *beta, int k2p2)
{
  // Grid: (m, 1, 1)
  // Block: (k2p2, 1, 1)

  if (threadIdx.x >= k2p2) { return; }

  float *inv = &Xinv[blockIdx.x * (k2p2 * k2p2)];
  float *vct = &beta0[blockIdx.x * k2p2];
  float accum = 0.0;

  for (int i = 0; i < k2p2; i++) {
    accum += inv[IDX_2D(threadIdx.x, i, k2p2)] * vct[i];
  }

  beta[IDX_2D(blockIdx.x, threadIdx.x, blockDim.x)] = accum;
}

extern "C" void bfast_step_4b_single(float *Xinv, float *beta0, float **beta,
    int m, int k2p2)
{
  float *d_Xinv = NULL, *d_beta0 = NULL, *d_beta = NULL;
  const size_t mem_Xinv = m * k2p2 * k2p2 * sizeof(float);
  const size_t mem_beta0 = m * k2p2 * sizeof(float);
  const size_t mem_beta = mem_beta0;

  CUDA_SUCCEED(cudaMalloc(&d_Xinv, mem_Xinv));
  CUDA_SUCCEED(cudaMalloc(&d_beta0, mem_beta0));
  CUDA_SUCCEED(cudaMalloc(&d_beta, mem_beta));

  CUDA_SUCCEED(cudaMemcpy(d_Xinv, Xinv, mem_Xinv, cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_beta0, beta0, mem_beta0, cudaMemcpyHostToDevice));

  dim3 block(8, 1, 1); // Assumes k2p2 <= 8
  dim3 grid(m, 1, 1);
  bfast_step_4b<<<grid, block>>>(d_Xinv, d_beta0, d_beta, k2p2);

  *beta = (float *)malloc(mem_beta);
  CUDA_SUCCEED(cudaMemcpy(*beta, d_beta, mem_beta, cudaMemcpyDeviceToHost));
  CUDA_SUCCEED(cudaFree(d_Xinv));
  CUDA_SUCCEED(cudaFree(d_beta0));
  CUDA_SUCCEED(cudaFree(d_beta));
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Step 4c: Calculating y_preds
//
// Input:
//   X:       [k2p2][N]f32
//   beta:    [m][k2p2]f32
// Output:
//   y_preds: [m][N]f32
//
// Similar reasoning as in 4a. Consider merging these two kernels, the only
// difference is the filtering.
//
// Perform matrix-matrix multiplication between X and beta.

__global__ void bfast_step_4c(float *X, float *beta, float *y_preds,
    int N, int m, int k2p2)
{
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;

  if(gidy >= m || gidx >= N) {
    return;
  }

  float accum = 0.0;
  for(int k = 0; k < k2p2; k ++) {
    accum += beta[IDX_2D(gidy, k, k2p2)] * X[IDX_2D(k, gidx, N)];
  }

  y_preds[IDX_2D(gidy, gidx, N)] = accum;
}

extern "C" void bfast_step_4c_single(float *Xt, float *beta, float **y_preds,
    int m, int N, int k2p2)
{
  // XXX: This function should actually take X as input, not Xt!
  float *d_Xt = NULL, *d_beta = NULL, *d_X = NULL;
  float *d_y_preds = NULL;
  const size_t mem_Xt = N * k2p2 * sizeof(float);
  const size_t mem_beta = m * k2p2 * sizeof(float);
  const size_t mem_y_preds = m * N * sizeof(float);

  CUDA_SUCCEED(cudaMalloc(&d_X, mem_Xt));
  CUDA_SUCCEED(cudaMalloc(&d_Xt, mem_Xt));
  CUDA_SUCCEED(cudaMalloc(&d_beta, mem_beta));
  CUDA_SUCCEED(cudaMalloc(&d_y_preds, mem_y_preds));

  CUDA_SUCCEED(cudaMemcpy(d_Xt, Xt, mem_Xt, cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_beta, beta, mem_beta, cudaMemcpyHostToDevice));

  transpose(d_Xt, d_X, N, k2p2);

  dim3 block(16, 16, 1);
  dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(m, block.y), 1);
  bfast_step_4c<<<grid, block>>>(d_X, d_beta, d_y_preds, N, m, k2p2);


  *y_preds = (float *)malloc(mem_y_preds);
  CUDA_SUCCEED(cudaMemcpy(*y_preds, d_y_preds, mem_y_preds,
        cudaMemcpyDeviceToHost));
  CUDA_SUCCEED(cudaFree(d_X));
  CUDA_SUCCEED(cudaFree(d_Xt));
  CUDA_SUCCEED(cudaFree(d_beta));
  CUDA_SUCCEED(cudaFree(d_y_preds));
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Step 5: Calculating Nss, y_erros, val_indss
//
// Input:
//   Y:         [m][N]f32
//   y_preds:   [m][N]f32
// Output:
//   Nss:       [m]i32
//   y_errors:  [m][N]f32
//   val_indss: [m][N]i32
//

__global__ void bfast_step_5(float *Y, float *y_preds, int *Nss,
    float *y_errors, int *val_indss, int N)
{
  // Grid: (m, 1, 1)
  // Block: (1024, 1, 1)

  if (threadIdx.x >= N) { return; }

  float *y = &Y[blockIdx.x * N];
  float *y_pred = &y_preds[blockIdx.x * N];
  float *y_error = &y_errors[blockIdx.x * N];
  int *val_inds = &val_indss[blockIdx.x * N];
  int *Ns = &Nss[blockIdx.x];

  float val = y[threadIdx.x];
  float err = !isnan(val) ? val - y_pred[threadIdx.x] : NAN;

  // Partition
  __shared__ int num_valids[1024];
  num_valids[threadIdx.x] = !isnan(err);
  __syncthreads();
  scaninc_block_add<int>(num_valids);
  int i = num_valids[N - 1];

  unsigned int idx;
  if (!isnan(err)) {
    idx = num_valids[threadIdx.x] - 1;
  } else {
    float num_invalids = threadIdx.x - (num_valids[threadIdx.x] - 1);
    idx = num_invalids - 1 + i;
    //idx = threadIdx.x - num_valids[threadIdx.x] + i;
  }

  y_error[idx] = err;
  val_inds[idx] = threadIdx.x;
  if (threadIdx.x == 0) {
    *Ns = i;
  }
}

extern "C" void bfast_step_5_single(float *Y, float *y_preds, int **Nss,
    float **y_errors, int **val_indss, int N, int m)
{
  float *d_Y = NULL, *d_y_preds = NULL, *d_y_errors = NULL;
  int *d_Nss = NULL, *d_val_indss = NULL;
  const size_t mem_Y = m * N * sizeof(float);
  const size_t mem_y_preds = mem_Y;
  const size_t mem_Nss = m * sizeof(float);
  const size_t mem_y_errors = mem_Y;
  const size_t mem_val_indss = mem_Y;

  CUDA_SUCCEED(cudaMalloc(&d_Y, mem_Y));
  CUDA_SUCCEED(cudaMalloc(&d_y_preds, mem_y_preds));
  CUDA_SUCCEED(cudaMalloc(&d_Nss, mem_Nss));
  CUDA_SUCCEED(cudaMalloc(&d_y_errors, mem_y_errors));
  CUDA_SUCCEED(cudaMalloc(&d_val_indss, mem_val_indss));

  CUDA_SUCCEED(cudaMemcpy(d_Y, Y, mem_Y, cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_y_preds, y_preds, mem_y_preds, cudaMemcpyHostToDevice));

  dim3 block(1024, 1, 1);
  dim3 grid(m, 1, 1);
  bfast_step_5<<<grid, block>>>(d_Y, d_y_preds, d_Nss, d_y_errors, d_val_indss, N);

  *Nss = (int *)malloc(mem_Nss);
  *y_errors = (float *)malloc(mem_y_errors);
  *val_indss = (int *)malloc(mem_val_indss);
  CUDA_SUCCEED(cudaMemcpy(*Nss, d_Nss, mem_Nss, cudaMemcpyDeviceToHost));
  CUDA_SUCCEED(cudaMemcpy(*y_errors, d_y_errors, mem_y_errors,
        cudaMemcpyDeviceToHost));
  CUDA_SUCCEED(cudaMemcpy(*val_indss, d_val_indss, mem_val_indss,
        cudaMemcpyDeviceToHost));

  CUDA_SUCCEED(cudaFree(d_Y));
  CUDA_SUCCEED(cudaFree(d_y_preds));
  CUDA_SUCCEED(cudaFree(d_Nss));
  CUDA_SUCCEED(cudaFree(d_y_errors));
  CUDA_SUCCEED(cudaFree(d_val_indss));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Step 6: Calculating nss, sigmas
//
// Input:
//   Y:       [m][N]f32     (only using slice: [m][n])
//   y_preds: [m][N]f32
// Output:
//   nss:     [m]i32
//   sigmas:  [m]f32
//

__global__ void bfast_step_6(float *Yh, float *y_errors, int *nss,
    float *sigmas, int n, int N, int k2p2)
{
  // Grid dimensions (x, y, z): (m, 1, 1)
  // Block dimensions (x, y, z ): (1024, 1, 1)

  if (threadIdx.x >= n) { return; }

  float *yh = &Yh[blockIdx.x * N]; // Yh is Y, so N cols in memory
  float *y_error = &y_errors[blockIdx.x * N];

  __shared__ int num_valids[1024];
  num_valids[threadIdx.x] = !isnan(yh[threadIdx.x]);
  __syncthreads();
  scaninc_block_add<int>(num_valids);
  int ns = num_valids[n - 1];

  // hacky optimization: reuse num_valids by ptr cast
  // __shared__ float sigma_shared[1024]; 
  float *sigma_shared = (float *) &num_valids;
  float val = threadIdx.x < ns ? y_error[threadIdx.x] : 0.0;
  val = val * val;
  sigma_shared[threadIdx.x] = val;
  __syncthreads();
  scaninc_block_add<float>(sigma_shared);

  if (threadIdx.x == 0) {
    //float sigma0 = sigma_shared[n - 1];
    //float sigma = sqrtf(sigma0 / ((float)(ns - k2p2)));
    sigmas[blockIdx.x] = __fsqrt_rd(sigma_shared[n - 1] / ((float)(ns - k2p2)));
    nss[blockIdx.x] = ns;
  }
}

extern "C" void bfast_step_6_single(float *Y, float *y_errors,  int **nss,
    float **sigmas, int n, int k2p2, int m, int N)
{
  float *d_Y = NULL, *d_y_errors = NULL, *d_sigmas = NULL;
  int *d_nss = NULL;
  const size_t mem_Y = m * N * sizeof(float);
  const size_t mem_y_errors = mem_Y;
  const size_t mem_nss = m * sizeof(float);
  const size_t mem_sigmas = mem_nss;

  CUDA_SUCCEED(cudaMalloc(&d_Y, mem_Y));
  CUDA_SUCCEED(cudaMalloc(&d_y_errors, mem_y_errors));
  CUDA_SUCCEED(cudaMalloc(&d_nss, mem_nss));
  CUDA_SUCCEED(cudaMalloc(&d_sigmas, mem_sigmas));

  CUDA_SUCCEED(cudaMemcpy(d_Y, Y, mem_Y, cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_y_errors, y_errors, mem_y_errors,
        cudaMemcpyHostToDevice));

  fprintf(stderr, "n=%d, k2p2=%d, m=%d, N=%d\n", n, k2p2, m, N);
  dim3 block(1024, 1, 1);
  dim3 grid(m, 1, 1);
  bfast_step_6<<<grid, block>>>(d_Y, d_y_errors, d_nss, d_sigmas, n, N, k2p2);

  *nss = (int *)malloc(mem_nss);
  *sigmas = (float *)malloc(mem_sigmas);

  CUDA_SUCCEED(cudaMemcpy(*nss, d_nss, mem_nss, cudaMemcpyDeviceToHost));
  CUDA_SUCCEED(cudaMemcpy(*sigmas, d_sigmas, mem_sigmas,
        cudaMemcpyDeviceToHost));

  CUDA_SUCCEED(cudaFree(d_Y));
  CUDA_SUCCEED(cudaFree(d_y_errors));
  CUDA_SUCCEED(cudaFree(d_nss));
  CUDA_SUCCEED(cudaFree(d_sigmas));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Step 7a: Produces some interesting value.
//
// Input:
//    y_errors: [m][N]
//    nss:      [m]
// Output:
//    MO_fsts:  [m]

__global__ void bfast_step_7a(float *y_errors,
                                int *nss,
                                int  h,
                                int  N,
                              float *MO_fsts)
{
  // Grid:  (m, 1, 1)
  // Block: (1024, 1, 1)

  if (h <= threadIdx.x) { return; }

  float *y_error = &y_errors[blockIdx.x * N];
  float *MO_fst  = &MO_fsts [blockIdx.x];
  int    ns      = nss      [blockIdx.x];

  __shared__ float errs[1024];

  errs[threadIdx.x] = y_error[threadIdx.x  + ns - h + 1];
  __syncthreads();

  scaninc_block_add(errs);

  if (threadIdx.x == 0) {
    *MO_fst = errs[h-1];
  }
}

extern "C" void 
bfast_step_7a_single(float  *y_errors,
                       int  *nss,
                       int   h,
                       int   N,
                       int   m,
                     float **MO_fsts)
{
  float *d_y_errors = NULL;
  int   *d_nss      = NULL;
  float *d_MO_fsts  = NULL;

  const size_t mem_y_errors = m * N * sizeof(float);
  const size_t mem_nss      = m * sizeof(float);
  const size_t mem_MO_fsts  = m * sizeof(float);

  CUDA_SUCCEED(cudaMalloc(&d_y_errors, mem_y_errors));
  CUDA_SUCCEED(cudaMalloc(&d_nss, mem_nss));
  CUDA_SUCCEED(cudaMalloc(&d_MO_fsts, mem_MO_fsts));

  CUDA_SUCCEED(cudaMemcpy(d_y_errors, y_errors, mem_y_errors, cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_nss, nss, mem_nss, cudaMemcpyHostToDevice));

  fprintf(stderr, "h=%d, N=%d, m=%d", h, N, m);

  dim3 grid(m, 1, 1);
  dim3 block(1024, 1, 1);
  bfast_step_7a<<<grid, block>>>(d_y_errors, d_nss, h, N, d_MO_fsts);

  *MO_fsts = (float *)malloc(mem_MO_fsts);
  CUDA_SUCCEED(cudaMemcpy(*MO_fsts, d_MO_fsts, mem_MO_fsts, cudaMemcpyDeviceToHost));

  CUDA_SUCCEED(cudaFree(d_MO_fsts));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Step 7b: Produces a BOUND value of at least lam for each step in the monitor period. 
//
// Input:
//    lam:   0
//    n:     0
//    N:     0
// Output:
//    BOUND: [N-n]

__global__ void bfast_step_7b(float lam,
                              int   n,
                              int   N,
                            float  *BOUND)
{
  // Grid: (1, 1, 1)
  // Block: (1024, 1, 1)

  // int monitor_period_sz = N-n;
  if ( threadIdx.x < N-n ) {

    // Index into monitor period
    //unsigned int t = n + 1 + threadIdx.x;

    float frac = fdividef(n + 1 + threadIdx.x, n);

    /*
    // logplus(frac). Assures `tmp` is at least 1.
    if (frac > __expf(1.0f)) { BOUND[threadIdx.x] = lam * __fsqrt_rd( __logf(frac)); }
    else                     { BOUND[threadIdx.x] = lam; }
    */

    //BOUND[threadIdx.x] = lam * ( frac>__expf(1.0f) ? __fsqrt_rd(__logf(frac)) : 1);
    BOUND[threadIdx.x] = lam * ( frac>expf(1.0f) ? sqrtf(logf(frac)) : 1);

  }
}

extern "C" void bfast_step_7b_single(float lam, int n, int N, float
    **BOUND)
{
  float *d_BOUND = NULL;

  const size_t mem_BOUND = (N - n)  * sizeof(float);
  
  CUDA_SUCCEED(cudaMalloc(&d_BOUND, mem_BOUND));

  CUDA_SUCCEED(cudaMemcpy(d_BOUND, BOUND, mem_BOUND, cudaMemcpyHostToDevice));

  fprintf(stderr, "lam=%f, n=%d, N=%d\n", lam, n, N);

  dim3 grid(1, 1, 1);
  dim3 block(1024, 1, 1);
  bfast_step_7b<<<grid, block>>>(lam, n, N, d_BOUND);

  *BOUND = (float *)malloc(mem_BOUND);

  CUDA_SUCCEED(cudaMemcpy(*BOUND, d_BOUND, mem_BOUND, cudaMemcpyDeviceToHost));

  CUDA_SUCCEED(cudaFree(d_BOUND));
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Step 8: Mapping each sample to its excess value w.r.t. its bound. Maybe.
//
// Input:
//   y_errors[]:  [m][N]
//   val_indss[]: [m][N]
//   Nss[]:       [m]
//   nss[]:       [m]
//   sigmas[]:    [m]
//   MO_fsts[]:   [m]
//   BOUND[]:     [N-n]
//   h:
//   m:
//   N:
// Output:
//   breakss[]:   [m][N-n]

__global__ void bfast_step_8(float *y_errors,  // [m][N]
                               int *val_indss, // [m][N]
                               int *Nss,       // [m]
                               int *nss,       // [m]
                             float *sigmas,    // [m]
                             float *MO_fsts,   // [m]
                             float *BOUND,     // [N-n]
                               int h,
                               int n,
                               int N,
                             float *breakss)   // [m][N-n] output
{
  // Layout:
  // Grid:  (m, 1, 1)
  // Block: (1024, 1, 1)

  // Optimization opportunities:
  // Done. Read bound into shared memory.
  // Reuse shared memory for MO MOP MOPP
  // Reuse threadIdx.x instead of copying to local variable.

  if (threadIdx.x >= N-n) { return; }

  // In order of appearence
  int   Ns        = Nss       [blockIdx.x];
  int   ns        = nss       [blockIdx.x];
  float sigma     = sigmas    [blockIdx.x];
  float MO_fst    = MO_fsts   [blockIdx.x];
  float *y_error  = &y_errors [blockIdx.x * N];
  int   *val_inds = &val_indss[blockIdx.x * N];
  float *breaks   = &breakss  [blockIdx.x * (N-n)];
  float val;


  __shared__ float BOUND_shr[1024];
  
  if (threadIdx.x < N) {
    BOUND_shr[threadIdx.x] = BOUND[threadIdx.x];
  }

  __shared__ int val_inds_shr[1024];
  
  if (threadIdx.x < N) {
    val_inds_shr[threadIdx.x] = val_inds[threadIdx.x];
  }

  __shared__ float MO_shr[1024];
  {
    if      ( Ns-ns       <= threadIdx.x ) { MO_shr[threadIdx.x] = 0.0f;   }
    else if ( threadIdx.x == 0           ) { MO_shr[threadIdx.x] = MO_fst; }
    else                   { MO_shr[threadIdx.x] = -y_error[ns - h + threadIdx.x] 
                                                  + y_error[ns + threadIdx.x]; }
    __syncthreads();
    scaninc_block_add<float>(MO_shr);
  }

  {
    // MO'
    __syncthreads();
    //MO_shr[threadIdx.x] = fdividef( MO_shr[threadIdx.x] , sigma * __fsqrt_rd( (float)ns ));
    MO_shr[threadIdx.x] = fdividef( MO_shr[threadIdx.x] , sigma ) * rsqrtf( (float)ns );
  }

  {
    __syncthreads();

    if ( threadIdx.x < Ns - ns ) {
      val = MO_shr[threadIdx.x];
    }
    else {
      val = NAN;
    }

    // Make sure all threads has read into `val` before overwriting source.
    __syncthreads();
    MO_shr[val_inds_shr[threadIdx.x + ns] - n] = val;
  }

  // Here might be a producer/consumer dependency in MO_shr.

  {
    // breaks = ..
    __syncthreads();
    float m = MO_shr   [threadIdx.x];
    float b = BOUND_shr[threadIdx.x];

    if (isnan(m) || isnan(b)) { breaks[threadIdx.x] = 0.0f; }
    else                      { breaks[threadIdx.x] = fabsf(m) - b; }
  }

}




extern "C" void
bfast_step_8_single(float  *y_errors,  // [m][N]
                      int  *val_indss, // [m][N]
                      int  *Nss,       // [m]
                      int  *nss,       // [m]
                    float  *sigmas,    // [m]
                    float  *MO_fsts,   // [m]
                    float  *BOUND,     // [N-n]
                      int  h,
                      int  m,
                      int  N,
                      int  n,
                    float **breakss)   // [m][N-n]
{

  float *d_y_errors  = NULL;
  int   *d_val_indss = NULL;
  int   *d_Nss       = NULL;
  int   *d_nss       = NULL;
  float *d_sigmas    = NULL;
  float *d_MO_fsts   = NULL;
  float *d_BOUND     = NULL;
  float *d_breakss = NULL;

  const size_t mem_y_errors  = m * N * sizeof(float);
  const size_t mem_val_indss = m * N * sizeof(int);
  const size_t mem_Nss       = m     * sizeof(int);
  const size_t mem_nss       = m     * sizeof(int);
  const size_t mem_sigmas    = m     * sizeof(float);
  const size_t mem_MO_fsts   = m     * sizeof(float);
  const size_t mem_BOUND     =     N * sizeof(float);

  const size_t mem_breakss   = m * N * sizeof(float);

  CUDA_SUCCEED(cudaMalloc(&d_y_errors,  mem_y_errors));
  CUDA_SUCCEED(cudaMalloc(&d_val_indss, mem_val_indss));
  CUDA_SUCCEED(cudaMalloc(&d_Nss,       mem_Nss));
  CUDA_SUCCEED(cudaMalloc(&d_nss,       mem_nss));
  CUDA_SUCCEED(cudaMalloc(&d_sigmas,    mem_sigmas));
  CUDA_SUCCEED(cudaMalloc(&d_MO_fsts,   mem_MO_fsts));
  CUDA_SUCCEED(cudaMalloc(&d_BOUND,     mem_BOUND));
  CUDA_SUCCEED(cudaMalloc(&d_breakss,     mem_breakss));

  CUDA_SUCCEED(cudaMemcpy(d_y_errors,  y_errors,  mem_y_errors,  cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_val_indss, val_indss, mem_val_indss, cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_Nss,       Nss,       mem_Nss,       cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_nss,       nss,       mem_nss,       cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_sigmas,    sigmas,    mem_sigmas,    cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_MO_fsts,   MO_fsts,   mem_MO_fsts,   cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_BOUND,     BOUND,     mem_BOUND,     cudaMemcpyHostToDevice));

  fprintf(stderr, "h=%d, m=%d, n=%d\n", h, m, n);

  dim3 grid(m, 1, 1);
  dim3 block(1024, 1, 1);
  bfast_step_8<<<grid, block>>>
  (d_y_errors, d_val_indss, d_Nss, d_nss, d_sigmas, d_MO_fsts, d_BOUND, h, n, N, d_breakss);

  *breakss = (float *)malloc(mem_breakss);
  CUDA_SUCCEED(cudaMemcpy(*breakss, d_breakss, mem_breakss, cudaMemcpyDeviceToHost));

  CUDA_SUCCEED(cudaFree(d_MO_fsts));
  CUDA_SUCCEED(cudaFree(d_y_errors));
  CUDA_SUCCEED(cudaFree(d_val_indss));
  CUDA_SUCCEED(cudaFree(d_Nss));
  CUDA_SUCCEED(cudaFree(d_nss));
  CUDA_SUCCEED(cudaFree(d_sigmas));
  CUDA_SUCCEED(cudaFree(d_BOUND));
  CUDA_SUCCEED(cudaFree(d_breakss));
}

extern "C" void bfast_naive(struct bfast_in *in, struct bfast_out *out)
{
  int k = in->k;
  int n = in->n;
  float f = in->freq;
  float hfrac = in->hfrac;
  float lam = in->lam;
  float *Y = in->images;
  const int m = in->shp[0];
  const int N = in->shp[1];

  int k2p2 = k * 2 + 2;
  int h = (int) ((float)n * hfrac);

  float *d_Y, *d_X, *d_Xt, *d_Xsqr, *d_Xinv, *d_Yt;
  float *d_beta0, *d_beta0t, *d_beta, *d_betat, *d_y_preds, *d_y_predst;
  int *d_Nss, *d_val_indss, *d_nss;
  float *d_sigmas, *d_MO_fsts, *d_y_errors, *d_BOUND, *d_breakss;

  const size_t mem_X = k2p2 * N * sizeof(float);
  const size_t mem_Y = m * N * sizeof(float);
  const size_t mem_Xsqr = m * k2p2 * k2p2 * sizeof(float);
  const size_t mem_Xinv = m * k2p2 * k2p2 * sizeof(float);
  const size_t mem_beta0 = m * k2p2 * sizeof(float);
  const size_t mem_beta = m * k2p2 * sizeof(float);
  const size_t mem_y_preds = m * N * sizeof(float);
  const size_t mem_Nss = m * sizeof(int);
  const size_t mem_y_errors = mem_Y;
  const size_t mem_val_indss = mem_Y;
  const size_t mem_nss = m * sizeof(int);
  const size_t mem_sigmas = mem_nss;
  const size_t mem_MO_fsts = mem_nss;
  const size_t mem_BOUND = (N - n) * sizeof(float);
  const size_t mem_breakss = m * (N - n) * sizeof(float);

  CUDA_SUCCEED(cudaMalloc(&d_X, mem_X));
  CUDA_SUCCEED(cudaMalloc(&d_Xt, mem_X));
  CUDA_SUCCEED(cudaMalloc(&d_Y, mem_Y));
  CUDA_SUCCEED(cudaMalloc(&d_Xsqr, mem_Xsqr));
  CUDA_SUCCEED(cudaMalloc(&d_Xinv, mem_Xinv));
  CUDA_SUCCEED(cudaMalloc(&d_Yt, mem_Y));
  CUDA_SUCCEED(cudaMalloc(&d_beta0, mem_beta0));
  CUDA_SUCCEED(cudaMalloc(&d_beta0t, mem_beta0));
  CUDA_SUCCEED(cudaMalloc(&d_beta, mem_beta));
  CUDA_SUCCEED(cudaMalloc(&d_betat, mem_beta));
  CUDA_SUCCEED(cudaMalloc(&d_y_preds, mem_y_preds));
  CUDA_SUCCEED(cudaMalloc(&d_y_predst, mem_y_preds));
  CUDA_SUCCEED(cudaMalloc(&d_Nss, mem_Nss));
  CUDA_SUCCEED(cudaMalloc(&d_y_errors, mem_y_errors));
  CUDA_SUCCEED(cudaMalloc(&d_val_indss, mem_val_indss));
  CUDA_SUCCEED(cudaMalloc(&d_nss, mem_nss));
  CUDA_SUCCEED(cudaMalloc(&d_sigmas, mem_sigmas));
  CUDA_SUCCEED(cudaMalloc(&d_MO_fsts, mem_MO_fsts));
  CUDA_SUCCEED(cudaMalloc(&d_BOUND, mem_BOUND));
  CUDA_SUCCEED(cudaMalloc(&d_breakss, mem_breakss));

  CUDA_SUCCEED(cudaMemcpy(d_Y, Y, mem_Y, cudaMemcpyHostToDevice));

  CUDA_SUCCEED(cudaDeviceSynchronize());

  struct timer bfast_timer;
  struct timer kernel_timer[11];
  timer_reset(&bfast_timer);
  for (int i = 0; i < sizeof(kernel_timer)/sizeof(kernel_timer[0]); i++) {
    timer_reset(&kernel_timer[i]);
  }

  for (int i = 0; i < num_runs; i++) {
    if (!print_individual) { timer_start(&bfast_timer); }

    {
      timer_individual_start(kernel_timer, 0);
      dim3 block(16, 16, 1);
      dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(k2p2, block.y), 1);
      bfast_step_1<<<grid, block>>>(d_X, k2p2, N, f);
      timer_individual_stop(kernel_timer, 0);
    }

    {
      timer_individual_start(kernel_timer, 1);
      transpose(d_X, d_Xt, k2p2, N);
      transpose(d_Y, d_Yt, m, N);
      dim3 block(8, 8, 1); // Assumes k2p2 <= 8
      dim3 grid(CEIL_DIV(m, STEP_2_TILE_SIZE), 1, 1);
      bfast_step_2<<<grid, block>>>(d_X, d_Xt, d_Yt, d_Xsqr, N, n, k2p2, m);
      timer_individual_stop(kernel_timer, 1);
    }

    {
      timer_individual_start(kernel_timer, 2);
      dim3 block(16, 8, 1); // Assumes k2p2 <= 8
      dim3 grid(m, 1, 1);
      const size_t shared_size = k2p2 * 2 * k2p2 * sizeof(float);
      bfast_step_3<<<grid, block, shared_size>>>(d_Xsqr, d_Xinv, k2p2);
      timer_individual_stop(kernel_timer, 2);
    }

    {
      timer_individual_start(kernel_timer, 3);
      dim3 block(16, 16, 1);
      dim3 grid(CEIL_DIV(m, block.x), CEIL_DIV(k2p2, block.y), 1);
      bfast_step_4a<<<grid, block>>>(d_X, d_Yt, d_beta0t, k2p2, n, m, N);
      transpose(d_beta0t, d_beta0, k2p2, m);
      timer_individual_stop(kernel_timer, 3);
    }

    {
      timer_individual_start(kernel_timer, 4);
      dim3 block(8, 1, 1); // Assumes k2p2 <= 8
      dim3 grid(m, 1, 1);
      bfast_step_4b<<<grid, block>>>(d_Xinv, d_beta0, d_beta, k2p2);
      timer_individual_stop(kernel_timer, 4);
    }

    {
      timer_individual_start(kernel_timer, 5);
      dim3 block(16, 16, 1);
      dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(m, block.y), 1);
      bfast_step_4c<<<grid, block>>>(d_X, d_beta, d_y_preds, N, m, k2p2);
      timer_individual_stop(kernel_timer, 5);
    }

    {
      timer_individual_start(kernel_timer, 6);
      dim3 block(1024, 1, 1);
      dim3 grid(m, 1, 1);
      bfast_step_5<<<grid, block>>>(d_Y, d_y_preds, d_Nss, d_y_errors, d_val_indss, N);
      timer_individual_stop(kernel_timer, 6);
    }

    {
      timer_individual_start(kernel_timer, 7);
      dim3 block(1024, 1, 1);
      dim3 grid(m, 1, 1);
      bfast_step_6<<<grid, block>>>(d_Y, d_y_errors, d_nss, d_sigmas, n, N, k2p2);
      timer_individual_stop(kernel_timer, 7);
    }

    {
      timer_individual_start(kernel_timer, 8);
      dim3 block(1024, 1, 1);
      dim3 grid(m, 1, 1);
      bfast_step_7a<<<grid, block>>>(d_y_errors, d_nss, h, N, d_MO_fsts);
      timer_individual_stop(kernel_timer, 8);
    }

    {
      timer_individual_start(kernel_timer, 9);
      dim3 block(1024, 1, 1);
      dim3 grid(1, 1, 1);
      bfast_step_7b<<<grid, block>>>(lam, n, N, d_BOUND);
      timer_individual_stop(kernel_timer, 9);
    }

    {
      timer_individual_start(kernel_timer, 10);
      dim3 block(1024, 1, 1);
      dim3 grid(m, 1, 1);
      bfast_step_8<<<grid, block>>>(d_y_errors, d_val_indss, d_Nss, d_nss,
          d_sigmas, d_MO_fsts, d_BOUND, h, n, N, d_breakss);
      timer_individual_stop(kernel_timer, 10);
    }

    if (!print_individual) { timer_stop(&bfast_timer); }
  }

  if (print_individual) {
    for (int i = 0; i < sizeof(kernel_timer)/sizeof(kernel_timer[0]); i++) {
      const char *kernel_name;
      switch (i) {
      case 0:   kernel_name = "bfast_step_1";  break;
      case 1:   kernel_name = "bfast_step_2";  break;
      case 2:   kernel_name = "bfast_step_3";  break;
      case 3:   kernel_name = "bfast_step_4a"; break;
      case 4:   kernel_name = "bfast_step_4b"; break;
      case 5:   kernel_name = "bfast_step_4c"; break;
      case 6:   kernel_name = "bfast_step_5";  break;
      case 7:   kernel_name = "bfast_step_6";  break;
      case 8:   kernel_name = "bfast_step_7a"; break;
      case 9:   kernel_name = "bfast_step_7b"; break;
      case 10:  kernel_name = "bfast_step_8";  break;
      default:  assert(0);
      }
      timer_report(&kernel_timer[i], kernel_name);
    }
  } else {
    timer_report(&bfast_timer, "bfast");
  }

  out->breakss = (float *)malloc(m * (N - n) * sizeof(float));
  out->breakss[0] = 0.0;
  out->shp[0] = m;
  out->shp[1] = N - n;
  CUDA_SUCCEED(cudaMemcpy(out->breakss, d_breakss, mem_breakss, cudaMemcpyDeviceToHost));

  CUDA_SUCCEED(cudaFree(d_X));
  CUDA_SUCCEED(cudaFree(d_Xt));
  CUDA_SUCCEED(cudaFree(d_Y));
  CUDA_SUCCEED(cudaFree(d_Xsqr));
  CUDA_SUCCEED(cudaFree(d_Xinv));
  CUDA_SUCCEED(cudaFree(d_Yt));
  CUDA_SUCCEED(cudaFree(d_beta0));
  CUDA_SUCCEED(cudaFree(d_beta0t));
  CUDA_SUCCEED(cudaFree(d_beta));
  CUDA_SUCCEED(cudaFree(d_betat));
  CUDA_SUCCEED(cudaFree(d_y_preds));
  CUDA_SUCCEED(cudaFree(d_y_predst));
  CUDA_SUCCEED(cudaFree(d_Nss));
  CUDA_SUCCEED(cudaFree(d_y_errors));
  CUDA_SUCCEED(cudaFree(d_val_indss));
  CUDA_SUCCEED(cudaFree(d_nss));
  CUDA_SUCCEED(cudaFree(d_sigmas));
  CUDA_SUCCEED(cudaFree(d_MO_fsts));
  CUDA_SUCCEED(cudaFree(d_BOUND));
  CUDA_SUCCEED(cudaFree(d_breakss));
}
