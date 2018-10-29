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

