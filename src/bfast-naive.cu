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


class TupleInt {
  public:
    int x;
    int y;
    __device__ __host__ inline TupleInt()
    {
        x = 0; y = 0;
    }
    __device__ __host__ inline TupleInt(const int& a, const int& b)
    {
        x = a; y = b;
    }
    __device__ __host__ inline TupleInt(const TupleInt& i2)
    {
        x = i2.x; y = i2.y;
    }
    volatile __device__ __host__ inline TupleInt& operator=(const TupleInt& i2) volatile
    {
        x = i2.x; y = i2.y;
        return *this;
    }
};




TupleInt map_fun(float val)
{
  int nan = isnan(val);
  return TupleInt(!nan, nan);
}

TupleInt op(TupleInt a, TupleInt b)
{
  return TupleInt(a.x + b.x, a.y + b.y);
}

TupleInt scaninc_warp(volatile TupleInt *p)
{
  const unsigned int idx = threadIdx.x
  const unsigned int lane = idx & 31;

  // no synchronization needed inside a WARP,
  //   i.e., SIMD execution
  if (lane >= 1)  p[idx] = op(p[idx-1],  p[idx]);
  if (lane >= 2)  p[idx] = op(p[idx-2],  p[idx]);
  if (lane >= 4)  p[idx] = op(p[idx-4],  p[idx]);
  if (lane >= 8)  p[idx] = op(p[idx-8],  p[idx]);
  if (lane >= 16) p[idx] = op(p[idx-16], p[idx]);

  return const_cast<T&>(p[idx]);
}

TupleInt scaninc_map_warp(volatile float *in, volatile TupleInt *p)
{
  const unsigned int idx = threadIdx.x
  const unsigned int lane = idx & 31;

  // no synchronization needed inside a WARP,
  //   i.e., SIMD execution
  if (lane >= 1)  p[idx] = op(map_fun(in[idx-1]),  map_fun(in[idx]));
  if (lane >= 2)  p[idx] = op(map_fun(in[idx-2]),  map_fun(in[idx]));
  if (lane >= 4)  p[idx] = op(map_fun(in[idx-4]),  map_fun(in[idx]));
  if (lane >= 8)  p[idx] = op(map_fun(in[idx-8]),  map_fun(in[idx]));
  if (lane >= 16) p[idx] = op(map_fun(in[idx-16]), map_fun(in[idx]));

  return const_cast<T&>(p[idx]);
}

void scaninc_map_block(volatile float *in, volatile TupleInt *p)
{
  const unsigned int idx = threadIdx.x
  const unsigned int lane = idx &  31;
  const unsigned int warpid = idx >> 5;

  TupleInt val = scaninc_map_warp<OP>(in, p);
  __syncthreads();

  if (lane == 31) { p[warpid] = val; }
  __syncthreads();

  if (warpid == 0) scaninc_warp(p);
  __syncthreads();

  if (warpid > 0) {
      val = op(p[warpid-1], val);
  }

  __syncthreads();
  p[idx] = val;
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


float op2(float a, float b) { return a + b; }

float scaninc_warp_op2(volatile float *in, volatile float *p)
{
  const unsigned int idx = threadIdx.x
  const unsigned int lane = idx & 31;

  // no synchronization needed inside a WARP,
  //   i.e., SIMD execution
  if (lane >= 1)  p[idx] = op(in[idx-1],  in[idx]);
  if (lane >= 2)  p[idx] = op(in[idx-2],  in[idx]);
  if (lane >= 4)  p[idx] = op(in[idx-4],  in[idx]);
  if (lane >= 8)  p[idx] = op(in[idx-8],  in[idx]);
  if (lane >= 16) p[idx] = op(in[idx-16], in[idx]);

  return p[idx];
}

void scaninc_block_op2(volatile float *in, volatile float *p)
{
  const unsigned int idx = threadIdx.x
  const unsigned int lane = idx &  31;
  const unsigned int warpid = idx >> 5;

  float val = scaninc_map_warp(in, p);
  __syncthreads();

  if (lane == 31) { p[warpid] = val; }
  __syncthreads();

  if (warpid == 0) { scaninc_warp(p); }
  __syncthreads();

  if (warpid > 0) {
      val = op(p[warpid-1], val);
  }

  __syncthreads();
  p[idx] = val;
}

__global__ void step_6_kernel(float *Yh, float *y_errors,
    float *nss, float *sigmas,
    int m, int n, int N, int k2p2)
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
  float *d_beta0t; // k2p2 by m
  const size_t mem_beta0t = k2p2 * m * sizeof(float);
  {
    CUDA_SUCCEED(cudaMalloc(&d_beta0t, mem_beta0t));
    dim3 block(16, 16, 1);
    dim3 grid(CEIL_DIV(m, block.x),
              CEIL_DIV(k2p2, block.y),
              1);
    matmat_mul_filt<<<grid, block>>>(d_X, d_Yt, d_beta0t, k2p2, n, m);
  }

  float *d_beta0; // m by k2p2
  const size_t mem_beta0 = mem_beta0t;
  {
    CUDA_SUCCEED(cudaMalloc(&d_beta0, mem_beta0));
    dim3 block(16, 16, 1);
    dim3 grid(CEIL_DIV(m, block.x),
              CEIL_DIV(k2p2, block.y),
              1);
    mat_transpose<<<grid, block>>>(d_beta0t, d_beta0, k2p2, m);
  }


  float *d_beta; // m by k2p2
  const size_t mem_beta = mem_beta0;
  {
    CUDA_SUCCEED(cudaMalloc(&d_beta, mem_beta));
    dim3 block(16, 1, 1);
    dim3 grid(CEIL_DIV(k2p2, block.x),
              CEIL_DIV(m, block.y),
              1);
    block_matvecmul<<<grid, block>>>(d_Xinv, d_beta0, d_beta, k2p2);
  }


  float *d_y_preds; // m by N
  const size_t mem_y_preds = m * N * sizeof(float);
  {
    CUDA_SUCCEED(cudaMalloc(&d_y_preds, mem_y_preds));
    dim3 block(16, 16, 1);
    dim3 grid(CEIL_DIV(N, block.x),
              CEIL_DIV(m, block.y),
              1);
    matmat_mul<<<grid, block>>>(d_Xt, d_betat, d_y_preds, N, k2p2, m);
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
    step_5_kernel<<<grid, block>>>(d_Y, d_y_preds, d_Nss, d_y_errors,
        d_val_indss, N);
  }







  // Step 5: kernel for map body. blocksize=N


  // Step 6: kernel for map body. blocksize=n
  // Step 7: kernel for map body, kernel for map body
  // Step 8: kernel or map body, possibly scan




  out->breaks = NULL;
  out->shp[0] = 0;
  out->shp[1] = 0;

  //

}

