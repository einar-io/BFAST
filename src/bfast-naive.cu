#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "bfast.h"

#define CUDA_SUCCEED(x) (assert((x) == cudaSuccess))

#define IDX_2D(__r,__c,__nc) ((__r) * (__nc) + (__c))
#define CEIL_DIV(a,b) (((a) + (b) - 1) / (b))

__global__ void mk_X(float *X, int32_t k2p2, int32_t N, float f)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= k2p2 || j >= N) {
    return;
  }

  float val;
  if (i == 0) { val = 1.0; }
  else if (i == 1) { val = (float)j; }
  else {
    float angle = 2.0 * M_PI * (float)(i/2) * (float)j / f;
    if (i % 2 == 0) {
      val = sin(angle);
    } else {
      val = cos(angle);
    }
  }

  X[IDX_2D(i, j, N)] = val;
}

__global__ void mat_transpose(float *A, float *B, int heightA, int widthA)
{

  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;

  if(gidx >= widthA || gidy >= heightA) {
    return;
  }

  B[IDX_2D(gidx, gidy, heightA)] = A[IDX_2D(gidy, gidx, widthA)];
}

__global__ void block_matmat_filt(float *Xh, float *Xth, float *Yh,
    float *Xsqr, int k2p2, int n)
{
  // notation in the following 4 lines: __ is row by col
  // Xh is k2p2 by n
  // Xth is n by k2p2
  // Yh is m by n     (m is gridDim.x)
  // Xsqr is k2p2 by k2p2

  // Grid dimensions are (m, 1, 1)
  // Block dimensions are (k2p2, k2p2, 1)

  float *yh = &Yh[blockIdx.x * n];
  float accum = 0.0;

  if (threadIdx.y >= n || threadIdx.x >= k2p2) {
    return;
  }

  for (int k = 0; k < n; k++) {
    if (!isnan(yh[k])) {
      accum += Xh[IDX_2D(threadIdx.y, k, n)] *
                 Xth[IDX_2D(k, threadIdx.x, k2p2)];
    }
  }

  float *out_mat = &Xsqr[blockIdx.x * (k2p2 * k2p2)];
  out_mat[IDX_2D(threadIdx.y, threadIdx.x, k2p2)] = accum;
}

__global__ void block_mat_inv(float *Xsqr, float *Xinv, int k2p2)
{
  // m is gridDim.x
  // Xsqr shape: (m, k2p2, k2p2)
  // Xinv shape: (m, k2p2, k2p2)

  // Grid dimensions (x,y,z): (m, 1, 1)
  // Block dimensions (x,y,z): (2*k2p2, k2p2, 1)

  // When calling this kernel, remember to allocate dynamic shared memory:
  // block_mat_inv<<<blah, blah, k2p2*2*k2p2>>>(bla)

  if (threadIdx.x >= 2*k2p2 || threadIdx.y >= k2p2) {
    return;
  }

  float *sqr = &Xsqr[blockIdx.x * (k2p2 * k2p2)];
  float *inv = &Xinv[blockIdx.x * (k2p2 * k2p2)];

  extern __shared__ float A[]; // shape k2p2 by 2*k2p2

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

  // Body of guass_jordan map
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

__global__ void matmat_mul_filt(float *A, float *B, float *C, int rows_a,
    int cols_a, int cols_b)
{
  // When called in calculation of beta0 in bfast-distrib.fut:
  //    A (Xh) is k2p2 by n
  //    B (Yth) is n by m     (m is gridDim.x)
  //    C (beta0) is k2p2 by m
  // Spawn with one thread per element in the output matrix


  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;

  if(gidx >= cols_b || gidy >= rows_a) {
    return;
  }

  float accum = 0.0;
  for(int k = 0; k < cols_a; k ++) {
    float val = B[IDX_2D(k, gidx, cols_b)];
    if (!isnan(val)) {
      accum += A[IDX_2D(gidy, k, cols_a)] * val;
    }
  }

  C[IDX_2D(gidy, gidx, cols_b)] = accum;
}

__global__ void block_matvecmul(float *Xinv, float *beta0, float *beta, int k2p2)
{
  // The C output from matmat_mul_filt is as follows
  //   C =
  //     [
  //       [c_11, c_12, ..., c_1m], -- 1
  //       [c_21, c_22, ..., c_2m], -- 2
  //       ...
  //       [c_(k2p2)1, c_(k2p2)2, ..., c_(k2p2)m]  -- k2p2
  //     ]
  // We want beta0 to look as follows
  //   beta0 =
  //     [
  //       c_11, c_21, ..., c_(k2p2)1
  //       c_12, c_22, ..., c_(k2p2)2
  //       ...
  //       c_1m, c_2m, ..., c_(k2p2)m
  //     ]
  // From this, we see that transpose(C) is beta0

  // Xinv shape: (m, k2p2, k2p2)
  // beta0 is m by k2p2
  // beta is k2p2 by m

  // Each block calculates on matrix-vector multiplication
  // Block size is k2p2

  if (threadIdx.x >= k2p2) { return; }

  float *inv = &Xinv[blockIdx.x * (k2p2 * k2p2)];
  float *vct = &beta0[blockIdx.x * k2p2];
  float accum = 0.0;

  for (int i = 0; i < k2p2; i++) {
    accum += inv[IDX_2D(threadIdx.x, i, k2p2)] * vct[i];
  }

  beta[IDX_2D(threadIdx.x, blockIdx.x, blockDim.x)] = accum;
}

__global__ void matmat_mul(float *A, float *B, float *C, int rows_a,
    int cols_a, int cols_b)
{
  // When called in calculation of y_preds in bfast-distrib.fut:
  //    A (Xt) is N by k2p2
  //    B (beta) is k2p2 by m
  //    C (y_preds) N by m
  // Spawn with one thread per element in the output matrix


  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;

  if(gidx >= cols_b || gidy >= rows_a) {
    return;
  }

  float accum = 0.0;
  for(int k = 0; k < cols_a; k ++) {
    float val = B[IDX_2D(k, gidx, cols_b)];
    accum += A[IDX_2D(gidy, k, cols_a)] * val;
  }

  C[IDX_2D(gidy, gidx, cols_b)] = accum;
}

__global__ void step_5_kernel(float *images, float *y_preds, int *Nss,
    float *y_errors, int *val_indss, int N)
{
  // images is m by N
  // y_preds is m by N

  // Nss is m
  // y_errors is m by N
  // val_indss is m by N

  // Cosmin's tip: Assume N < 1024
  // I.e., each time series is handled within a block

  // Grid dimensions (x, y, z): (m, 1, 1)
  // Block dimensions (x, y, z ): (1024, 1, 1)

  /*
  if (threadIdx.x >= N) { return; }

  float *y = &images[blockIdx.x * gridDim.x];
  float *y_pred = &y_preds[blockIdx.x * gridDim.x];
  float *y_error = &y_errors[blockIdx.x * gridDim.x];
  __shared__ float y_error_all[1024];
  __shared__ int val_inds[1024];
  //int *Ns = &Nss[blockIdx.x];

  float val = y[threadIdx.x];
  if (!isnan(val)) {
    y_error_all[threadIdx.x] = val - y_pred[threadIdx.x];
  } else {
    y_error_all[threadIdx.x] = NAN;
  }
  */

  /*
     //  horribelt
  if (threadIdx.x == 0) {
    // blabla
    float *py = y_error;
    int *pi = val_inds;
    int ns = 0;
    for (int i = 0; i < N; i++) {
      float val = y_error_all[i];
      if (!isnan(val)) {
        *py = val;
        *pi = i;
        py++;
        pi++;
      } else {
        
      }
    }
  }
  */

}

extern "C" void bfast_naive(struct bfast_in *in, struct bfast_out *out)
{
  int k = in->k;
  int n = in->n;
  float freq = in->freq;
  float hfrac = in->hfrac;
  float lam = in->lam;
  float *images = in->images;
  const int m = in->shp[0];
  const int N = in->shp[1];


  int k2p2 = k*2+2;


  float *d_Y; // m by N
  const size_t mem_Y = m * N * sizeof(float);
  CUDA_SUCCEED(cudaMalloc(&d_Y, mem_Y));
  CUDA_SUCCEED(cudaMemcpy(d_Y, images, mem_Y, cudaMemcpyHostToDevice));


  float *d_X; // k2p2 by N
  const size_t mem_X = k2p2 * N * sizeof(float);
  {
    CUDA_SUCCEED(cudaMalloc(&d_X, mem_X));
    dim3 block(16, 16, 1);
    dim3 grid(CEIL_DIV(N, block.x),
              CEIL_DIV(k2p2, block.y),
              1);
    mk_X<<<grid, block>>>(d_X, k2p2, N, freq);
  }


  float *d_Xt; // N by k2p2
  const size_t mem_Xt = mem_X;
  {
    CUDA_SUCCEED(cudaMalloc(&d_Xt, mem_Xt));
    dim3 block(16, 16, 1);
    dim3 grid(CEIL_DIV(N, block.x),
              CEIL_DIV(k2p2, block.y),
              1);
    mat_transpose<<<grid, block>>>(d_X, d_Xt, k2p2, N);
  }


  float *d_Xsqr; // List of m matrices that are k2p2 by k2p2
  const size_t mem_Xsqr = k2p2 * k2p2;
  {
    CUDA_SUCCEED(cudaMalloc(&d_Xsqr, mem_Xsqr));
    dim3 block(8, 8, 1);
    dim3 grid(m, 1, 1);
    block_matmat_filt<<<grid, block>>>(d_X, d_Xt, d_Y, d_Xsqr, k2p2, n);
  }


  float *d_Xinv; // List of m matrices that are k2p2 by k2p2
  const size_t mem_Xinv = mem_Xsqr;
  {
    CUDA_SUCCEED(cudaMalloc(&d_Xinv, mem_Xinv));
    dim3 block(16, 8, 1);
    dim3 grid(m, 1, 1);
    const size_t shared_size = k2p2 * 2 * k2p2;
    block_mat_inv<<<grid, block, shared_size>>>(d_Xsqr, d_Xinv, k2p2);
  }


  float *d_Yt; // N by m
  const size_t mem_Yt = mem_Y;
  {
    CUDA_SUCCEED(cudaMalloc(&d_Yt, mem_Yt));
    dim3 block(16, 16, 1);
    dim3 grid(CEIL_DIV(N, block.x),
              CEIL_DIV(m, block.y),
              1);
    mat_transpose<<<grid, block>>>(d_Y, d_Yt, m, N);
  }


  // (k2p2 by n) times (n by m) is (k2p2 by m)
  float *d_beta0;
  const size_t mem_beta0 = k2p2 * m * sizeof(float);
  {
    CUDA_SUCCEED(cudaMalloc(&d_beta0, mem_beta0));
    dim3 block(16, 16, 1);
    dim3 grid(CEIL_DIV(m, block.x),
              CEIL_DIV(k2p2, block.y),
              1);
    matmat_mul_filt<<<grid, block>>>(d_X, d_Yt, d_beta0, k2p2, n, m);
  }


  //float *d_beta;
  //const size_t mem_beta = 
  //{

  //}



  // Step 1: mk_X, mat_transpose
  // Step 2: block_matmat_filt
  // Step 3: block_mat_inv
  // Step 4: transpose (for Yth), matmat_mul_filt, block_matvecmul, matmat_mul


  // Step 5: kernel for map body. blocksize=N


  // Step 6: kernel for map body. blocksize=n
  // Step 7: kernel for map body, kernel for map body
  // Step 8: kernel or map body, possibly scan




  out->breaks = NULL;
  out->shp[0] = 0;
  out->shp[1] = 0;

  //

}

