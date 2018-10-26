////////////////////////////////////////////////////////////////////////////////
//  Step 4c: Calculating y_preds
//
// Input:
//   X:       [k2p2][N]f32
//   beta:    [m][k2p2]f32
// Output:
//   y_preds: [m][N]f32
//
// Perform matrix-matrix multiplication between X and beta. Similar reasoning
// as in 4a, only difference between these two steps is the filtering.
#include "../bfast_util.cu.h"
#include "bfast_helpers.cu.h"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Naive implementation
//
// Calculates transpose(X)@transpose(beta) with filtering which equals
// transpose(y_preds)

__global__ void bfast_step_4c(float *Xt, float *betat, float *y_predst,
    int N, int m, int k2p2)
{
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;

  if(gidy >= N || gidx >= m) {
    return;
  }

  float accum = 0.0;
  for(int k = 0; k < k2p2; k ++) {
    accum += Xt[IDX_2D(gidy, k, k2p2)] * betat[IDX_2D(k, gidx, m)];
  }

  y_predst[IDX_2D(gidy, gidx, m)] = accum;
}

void bfast_step_4c_run(struct bfast_state *s)
{
  float *d_Xt = fget_dev_t(s,X), *d_betat = fget_dev_t(s,beta);
  float *d_y_predst = fget_dev_t(s,y_preds);
  int m = s->m, N = s->N, k2p2 = s->k2p2;

  dim3 block(16, 16, 1);
  dim3 grid(CEIL_DIV(m, block.x), CEIL_DIV(N, block.y), 1);
  bfast_step_4c<<<grid, block>>>(d_Xt, d_betat, d_y_predst, N, m, k2p2);
}

BFAST_BEGIN_TEST(bfast_step_4c_test)
  BFAST_BEGIN_INPUTS { BFAST_VALUE_X, BFAST_VALUE_beta } BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_y_preds } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS
  {
    BFAST_TRANSPOSE(X,transpose),
    BFAST_TRANSPOSE(beta,transpose),
    BFAST_STEP(bfast_step_4c_run),
    BFAST_UNTRANSPOSE(y_preds,transpose)
  }
  BFAST_END_STEPS
BFAST_END_TEST

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Flipped implementation
//
// Calculates beta@X with filtering which equals y_preds

__global__ void bfast_step_4c_flipped(float *X, float *beta, float *y_preds,
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

void bfast_step_4c_flipped_run(struct bfast_state *s)
{
  float *d_X = fget_dev(s,X), *d_beta = fget_dev(s,beta);
  float *d_y_preds = fget_dev(s,y_preds);
  int m = s->m, N = s->N, k2p2 = s->k2p2;

  dim3 block(16, 16, 1);
  dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(m, block.y), 1);
  bfast_step_4c_flipped<<<grid, block>>>(d_X, d_beta, d_y_preds, N, m, k2p2);
}


BFAST_BEGIN_TEST(bfast_step_4c_flipped_test)
  BFAST_BEGIN_INPUTS { BFAST_VALUE_X, BFAST_VALUE_beta } BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_y_preds } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS { BFAST_STEP(bfast_step_4c_flipped_run) } BFAST_END_STEPS
BFAST_END_TEST

