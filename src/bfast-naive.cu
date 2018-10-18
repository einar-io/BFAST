#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "bfast.h"
#include "bfast-helpers.cu.h"

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

__global__ void bfast_step_2(float *Xh, float *Xth, float *Yh, float *Xsqr,
    int N, int n, int k2p2)
{
  // Grid: (m, 1, 1)
  // Block: (k2p2, k2p2, 1)

  float *yh = &Yh[blockIdx.x * N];
  float accum = 0.0;

  if (threadIdx.y >= k2p2 || threadIdx.x >= k2p2) {
    return;
  }

  for (int k = 0; k < n; k++) {
    if (!isnan(yh[k])) {
      accum += Xh[IDX_2D(threadIdx.y, k, N)] *
                 Xth[IDX_2D(k, threadIdx.x, k2p2)];
    }
  }

  float *out_mat = &Xsqr[blockIdx.x * (k2p2 * k2p2)];
  out_mat[IDX_2D(threadIdx.y, threadIdx.x, k2p2)] = accum;
}

extern "C" void bfast_step_2_single(float *X, float *Xt, float *Y,
    float **Xsqr, int N, int n, int k2p2, int m)
{
  float *d_X = NULL, *d_Xt = NULL, *d_Y = NULL, *d_Xsqr = NULL;
  const size_t mem_X = k2p2 * N * sizeof(float);
  const size_t mem_Y = m * N * sizeof(float);
  const size_t mem_Xsqr = m * k2p2 * k2p2 * sizeof(float);

  CUDA_SUCCEED(cudaMalloc(&d_X, mem_X));
  CUDA_SUCCEED(cudaMalloc(&d_Xt, mem_X));
  CUDA_SUCCEED(cudaMalloc(&d_Y, mem_Y));
  CUDA_SUCCEED(cudaMalloc(&d_Xsqr, mem_Xsqr));

  CUDA_SUCCEED(cudaMemcpy(d_X, X, mem_X, cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_Xt, Xt, mem_X, cudaMemcpyHostToDevice));
  CUDA_SUCCEED(cudaMemcpy(d_Y, Y, mem_Y, cudaMemcpyHostToDevice));

  dim3 block(8, 8, 1); // Assumes k2p2 <= 8
  dim3 grid(m, 1, 1);
  bfast_step_2<<<grid, block>>>(d_X, d_Xt, d_Y, d_Xsqr, N, n, k2p2);

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

/*

__global__ void bfast_step_4b(float *Xinv, float *beta0, float *beta, int k2p2)
{
  // block_matvecmul
  // The C output from bfast_step_4a is as follows
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

__global__ void bfast_step_4c(float *A, float *B, float *C, int rows_a,
    int cols_a, int cols_b)
{
  // matmat_mul
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

__global__ void bfast_step_5(float *images, float *y_preds, int *Nss,
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

  if (threadIdx.x >= N) { return; }

  float *y = &images[blockIdx.x * N];
  float *y_pred = &y_preds[blockIdx.x * N];
  float *y_error = &y_errors[blockIdx.x * N];
  float *val_inds = &val_indss[blockIdx.x * N];
  int *Ns = &Nss[blockIdx.x];

  __shared__ float y_error_all[1024];

  float val = y[threadIdx.x];
  if (!isnan(val)) {
    y_error_all[threadIdx.x] = val - y_pred[threadIdx.x];
  } else {
    y_error_all[threadIdx.x] = NAN;
  }
  __syncthreads();


  //  Partition
  __shared__ TupleInt scan_res[1024];
  scaninc_map_block(y_error_all, scan_res);
  int i = scan_res[N - 1].x;

  __syncthreads();

  if (!isnan(y_error_all[threadIdx.x])) {
    unsigned int idx = scan_res[threadIdx.x].x - 1;
    y_error[idx] = y_error_all[threadIdx.x];
    val_inds[idx] = threadIdx.x;
  } else {
    unsigned int idx = scan_res[threadIdx.x].y - 1 + i;
    y_error[idx] = y_error_all[threadIdx.x];
    val_inds[idx] = threadIdx.x;
  }

  if (threadIdx.x == 0) {
    *Ns = i;
  }
}

__global__ void bfast_step_6(float *Yh, float *y_errors, float *nss,
    float *sigmas, int m, int n, int N, int k2p2)
{
  // Grid dimensions (x, y, z): (m, 1, 1)
  // Block dimensions (x, y, z ): (1024, 1, 1)


  if (threadIdx.x >= N) { return; }

  float *yh = &Yh[blockIdx.x * N]; // remember that Yh is m by N in memory (it's technically Y)
  float *y_error = &y_errors[blockIdx.x * N];

  __shared__ TupleInt scan_res[1024];
  if (threadIdx.x < n) {
    scaninc_map_block(yh, scan_res);
  }

  int ns = scan_res[n - 1].x;

  __shared__ float scan_me[1024];
  __shared__ float scan_res_2[1024];
  if (threadIdx.x < ns) {
    float val = y_error[threadIdx.x];
    scan_me[threadIdx.x] = val * val;
    scaninc_block_op2(scan_me, scan_res_2);
  }

  if (threadIdx.x == 0) {
    float sigma0 = scan_res_2[ns - 1];
    float sigma = sqrt(sigma / ((float)(ns-k2p2)));

    nss[blockIdx.x] = ns;
    sigmas[blockIdx.x] = sigma;
  }
}

extern "C" void bfast_step_naive(struct bfast_step_in *in,
    struct bfast_step_out *out)
{
  int k = in->k;
  int n = in->n;
  float freq = in->freq;
  float hfrac = in->hfrac;
  float lam = in->lam;
  float *images = in->images;
  const int m = in->shp[0];
  const int N = in->shp[1];

  int k2p2 = k * 2 + 2;


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
    bfast_step_1<<<grid, block>>>(d_X, k2p2, N, freq);
  }


  float *d_Xt; // N by k2p2
  const size_t mem_Xt = mem_X;
  {
    CUDA_SUCCEED(cudaMalloc(&d_Xt, mem_Xt));
    dim3 block(16, 16, 1);
    dim3 grid(CEIL_DIV(N, block.x),
              CEIL_DIV(k2p2, block.y),
              1);
    transpose_kernel<<<grid, block>>>(d_X, d_Xt, k2p2, N);
  }


  float *d_Xsqr; // List of m matrices that are k2p2 by k2p2
  const size_t mem_Xsqr = m * k2p2 * k2p2 * sizeof(float);
  {
    CUDA_SUCCEED(cudaMalloc(&d_Xsqr, mem_Xsqr));
    dim3 block(8, 8, 1);
    dim3 grid(m, 1, 1);
    bfast_step_2<<<grid, block>>>(d_X, d_Xt, d_Y, d_Xsqr, N, n, k2p2);
  }


  float *d_Xinv; // List of m matrices that are k2p2 by k2p2
  const size_t mem_Xinv = mem_Xsqr;
  {
    CUDA_SUCCEED(cudaMalloc(&d_Xinv, mem_Xinv));
    dim3 block(16, 8, 1);
    dim3 grid(m, 1, 1);
    const size_t shared_size = k2p2 * 2 * k2p2 * sizeof(float);
    bfast_step_3<<<grid, block, shared_size>>>(d_Xsqr, d_Xinv, k2p2);
  }


  float *d_Yt; // N by m
  const size_t mem_Yt = mem_Y;
  {
    CUDA_SUCCEED(cudaMalloc(&d_Yt, mem_Yt));
    dim3 block(16, 16, 1);
    dim3 grid(CEIL_DIV(N, block.x),
              CEIL_DIV(m, block.y),
              1);
    transpose_kernel<<<grid, block>>>(d_Y, d_Yt, m, N);
  }


  // (k2p2 by n) times (n by m) is (k2p2 by m)
  float *d_beta0t; // k2p2 by m
  const size_t mem_beta0t = k2p2 * m * sizeof(float);
  {
    CUDA_SUCCEED(cudaMalloc(&d_beta0t, mem_beta0t));
    dim3 block(16, 16, 1);
    dim3 grid(CEIL_DIV(m, block.x),
              CEIL_DIV(k2p2, block.y),
              1);
    bfast_step_4a<<<grid, block>>>(d_X, d_Yt, d_beta0t, k2p2, n, m, N);
  }

  float *d_beta0; // m by k2p2
  const size_t mem_beta0 = mem_beta0t;
  {
    CUDA_SUCCEED(cudaMalloc(&d_beta0, mem_beta0));
    dim3 block(16, 16, 1);
    dim3 grid(CEIL_DIV(m, block.x),
              CEIL_DIV(k2p2, block.y),
              1);
    transpose_kernel<<<grid, block>>>(d_beta0t, d_beta0, k2p2, m);
  }


  float *d_beta; // m by k2p2
  const size_t mem_beta = mem_beta0;
  {
    CUDA_SUCCEED(cudaMalloc(&d_beta, mem_beta));
    dim3 block(16, 1, 1);
    dim3 grid(CEIL_DIV(k2p2, block.x),
              CEIL_DIV(m, block.y),
              1);
    bfast_step_4b<<<grid, block>>>(d_Xinv, d_beta0, d_beta, k2p2);
  }


  float *d_y_preds; // m by N
  const size_t mem_y_preds = m * N * sizeof(float);
  {
    CUDA_SUCCEED(cudaMalloc(&d_y_preds, mem_y_preds));
    dim3 block(16, 16, 1);
    dim3 grid(CEIL_DIV(N, block.x),
              CEIL_DIV(m, block.y),
              1);
    bfast_step_4c<<<grid, block>>>(d_Xt, d_betat, d_y_preds, N, k2p2, m);
  }


  int *d_Nss, *d_val_indss;
  float *d_y_errors;
  const size_t mem_Nss = m * sizeof(int);
  const size_t mem_y_errors = m * N * sizeof(float);
  const size_t mem_val_indss = m * N * sizeof(int);
  {
    CUDA_SUCCEED(cudaMalloc(&d_Nss, mem_Nss));
    CUDA_SUCCEED(cudaMalloc(&d_y_errors, mem_y_errors));
    CUDA_SUCCEED(cudaMalloc(&d_val_indss, mem_val_indss));
    dim3 block(1024, 1, 1);
    dim3 grid(m, 1, 1);
    bfast_step_5<<<grid, block>>>(d_Y, d_y_preds, d_Nss, d_y_errors,
        d_val_indss, N);
  }

}
*/

