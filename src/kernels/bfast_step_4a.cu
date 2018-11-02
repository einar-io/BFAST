////////////////////////////////////////////////////////////////////////////////
//  Step 4a: Calculating beta0
//
// Input:
//   X:     [k2p2][N]f32    (only using slice: [k2p2][n])
//   Y:     [m][N]f32       (only using slice: [m][n])
// Output:
//   beta0: [m][k2p2]f32
//
// This calculation is performed by transposing Y (so its dimensions become
// [N][m]) and then applying (filtered) matrix-matrix multiplication.
// The output will need to be transposed again, since:
//      [k2p2][N] multiplied with [N][m] is [k2p2][m]
//
#include "../bfast_util.cu.h"
#include "bfast_helpers.cu.h"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Tiled implementation of the naive version (not the flipped)
//

template <int T>
__global__ void mmmult_regtiled_4a(float* A, float* B, float* C,
                                   int heightA, int widthB, int widthA, int N)
{
  __shared__ float Ash[T][T];
  float cs[T];

  unsigned int ii = blockDim.y * blockIdx.y;
  unsigned int jjj = blockDim.x * blockDim.x * blockIdx.x;
  unsigned int jj = jjj + threadIdx.y * blockDim.x;
  unsigned int j = jj + threadIdx.x;
  unsigned int col = j;

  #pragma unroll
  for (int i = 0; i < T; i++) { cs[i] = 0.0; }

  for (int kk = 0; kk < widthA; kk += T) {
    // Copy slice A[ii:ii+T, kk:kk+T]
    if (ii + threadIdx.y < heightA && kk + threadIdx.x < widthA) {
      Ash[threadIdx.y][threadIdx.x] =
        A[IDX_2D(ii + threadIdx.y, kk + threadIdx.x, N)];
    } else {
      Ash[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    for (int k = 0; k < T; k++) {
      float b;
      if (kk + k < widthA /* heightB */ && col < widthB) {
        b = B[IDX_2D(kk + k, col, widthB)];
      } else {
        b = 0.0;
      }
      bool is_number_b = !isnan(b); // hoisting
      #pragma unroll
      for (int i = 0; i < T; i++) {
        if (is_number_b) {
          cs[i] += b * Ash[i][k];
        }
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < T; i++) {
    if (col < widthB && ii + i < heightA) {
      C[IDX_2D(ii + i, col, widthB)] = cs[i];
    }
  }
}

void bfast_step_4a_tiled_run(struct bfast_state *s)
{
  float *d_X = fget_dev(s,X), *d_Yt = fget_dev_t(s,Y);
  float *d_beta0t = fget_dev_t(s,beta0);
  int m = s->m, k2p2 = s->k2p2, n = s->n, N = s->N;

  const int T = 16;
  dim3 block(T, T, 1);
  dim3 grid(CEIL_DIV(m, T*T), CEIL_DIV(k2p2, T), 1);
  mmmult_regtiled_4a<T><<<grid, block>>>(d_X, d_Yt, d_beta0t, k2p2, m, n, N);
}

BFAST_BEGIN_TEST(bfast_step_4a_tiled_test)
  BFAST_BEGIN_INPUTS
  { BFAST_VALUE_X, BFAST_VALUE_Y }
  BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_beta0 } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS
  {
    BFAST_TRANSPOSE(Y, transpose),
    BFAST_STEP(bfast_step_4a_tiled_run),
    BFAST_UNTRANSPOSE(beta0, transpose)
  }
  BFAST_END_STEPS
BFAST_END_TEST

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Flipped implementation
//
// This is actually slower due to uncoalesced access to Yh.
// If the rest of the algorithm as a whole does not need a transposed version
// of Y, this function could still improve the overall performance, since Y is
// a relatively expensive matrix to transpose (around ~1.3ms).

__global__ void bfast_step_4a_flipped(float *Yh, float *Xth, float *beta0,
    int m, int k2p2, int n, int N)
{
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;

  if(gidy >= m || gidx >= k2p2) {
    return;
  }

  float accum = 0.0;
  for(int k = 0; k < n; k ++) {
    float val = Yh[IDX_2D(gidy, k, N)];
    if (!isnan(val)) {
      accum += Xth[IDX_2D(k, gidx, k2p2)] * val;
    }
  }

  beta0[IDX_2D(gidy, gidx, k2p2)] = accum;
}

void bfast_step_4a_flipped_run(struct bfast_state *s)
{
  float *d_Xt = fget_dev_t(s,X), *d_Y = fget_dev(s,Y);
  float *d_beta0 = fget_dev(s,beta0);
  int m = s->m, k2p2 = s->k2p2, n = s->n, N = s->N;

  dim3 block(8, 32, 1);
  dim3 grid(CEIL_DIV(k2p2, block.x), CEIL_DIV(m, block.y), 1);
  bfast_step_4a_flipped<<<grid, block>>>(d_Y, d_Xt, d_beta0, m, k2p2, n, N);
}

BFAST_BEGIN_TEST(bfast_step_4a_flipped_test)
  BFAST_BEGIN_INPUTS
  { BFAST_VALUE_X, BFAST_VALUE_Y }
  BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_beta0 } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS
  {
    BFAST_TRANSPOSE(X, transpose),
    BFAST_STEP(bfast_step_4a_flipped_run)
  }
  BFAST_END_STEPS
BFAST_END_TEST

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Naive implementation

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

void bfast_step_4a_run(struct bfast_state *s)
{
  float *d_X = fget_dev(s,X), *d_Yt = fget_dev_t(s,Y);
  float *d_beta0t = fget_dev_t(s,beta0);
  int m = s->m, k2p2 = s->k2p2, n = s->n, N = s->N;

  dim3 block(32, 8, 1);
  dim3 grid(CEIL_DIV(m, block.x), CEIL_DIV(k2p2, block.y), 1);
  bfast_step_4a<<<grid, block>>>(d_X, d_Yt, d_beta0t, k2p2, n, m, N);
}

BFAST_BEGIN_TEST(bfast_step_4a_test)
  BFAST_BEGIN_INPUTS
  { BFAST_VALUE_X, BFAST_VALUE_Y }
  BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_beta0 } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS
  {
    BFAST_TRANSPOSE(Y, transpose),
    BFAST_STEP(bfast_step_4a_run),
    BFAST_UNTRANSPOSE(beta0, transpose)
  }
  BFAST_END_STEPS
BFAST_END_TEST

