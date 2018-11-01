////////////////////////////////////////////////////////////////////////////////
//  Step 2: Calculating Xsqr
//
// Input:
//   X:    [k2p2][N]f32    (only using slice: [k2p2][n])
//   Y:    [m][N]f32       (only using slice: [m][n])
// Output:
//   Xsqr: [m][k2p2][k2p2]f32
//
#include "../bfast_util.cu.h"
#include "bfast_helpers.cu.h"


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Step 2 shared mem

#define STEP_2_TILE_SIZE 28

__global__ void bfast_step_2_shr(float *Xh, float *Xth, float *Yh, float *Xsqr,
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
        ysh[ysh_idx] = Yh[IDX_2D(y_row, i, N)];
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

void bfast_step_2_shr_run(struct bfast_state *s)
{
  float *d_X = fget_dev(s,X), *d_Xt = fget_dev_t(s,X);
  float *d_Y = fget_dev(s,Y), *d_Xsqr = fget_dev(s,Xsqr);
  int m = s->m, N = s->N, n = s->n, k2p2 = s->k2p2;

  dim3 block(8, 8, 1); // Assumes k2p2 <= 8
  dim3 grid(CEIL_DIV(m, STEP_2_TILE_SIZE), 1, 1);
  bfast_step_2_shr<<<grid, block>>>(d_X, d_Xt, d_Y, d_Xsqr, N, n, k2p2, m);
}

BFAST_BEGIN_TEST(bfast_step_2_shr_test)
  BFAST_BEGIN_INPUTS { BFAST_VALUE_X, BFAST_VALUE_Y } BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_Xsqr } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS
  {
    BFAST_TRANSPOSE(X, transpose),
    BFAST_STEP(bfast_step_2_shr_run)
  }
  BFAST_END_STEPS
BFAST_END_TEST

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Tiled implementation


__global__ void bfast_step_2_tiled(float *Xh, float *Xth, float *Yth,
    float *Xsqr, int N, int n, int k2p2, int m)
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

void bfast_step_2_tiled_run(struct bfast_state *s)
{
  float *d_X = fget_dev(s,X), *d_Xt = fget_dev_t(s,X);
  float *d_Yt = fget_dev_t(s,Y), *d_Xsqr = fget_dev(s,Xsqr);
  int m = s->m, N = s->N, n = s->n, k2p2 = s->k2p2;

  dim3 block(8, 8, 1); // Assumes k2p2 <= 8
  dim3 grid(CEIL_DIV(m, STEP_2_TILE_SIZE), 1, 1);
  bfast_step_2_tiled<<<grid, block>>>(d_X, d_Xt, d_Yt, d_Xsqr, N, n, k2p2, m);
}

BFAST_BEGIN_TEST(bfast_step_2_tiled_test)
  BFAST_BEGIN_INPUTS { BFAST_VALUE_X, BFAST_VALUE_Y } BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_Xsqr } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS
  {
    BFAST_TRANSPOSE(X, transpose),
    BFAST_TRANSPOSE(Y, transpose),
    BFAST_STEP(bfast_step_2_tiled_run)
  }
  BFAST_END_STEPS
BFAST_END_TEST

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Naive implementation

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

void bfast_step_2_run(struct bfast_state *s)
{
  float *d_X = fget_dev(s,X), *d_Xt = fget_dev_t(s,X);
  float *d_Y = fget_dev(s,Y), *d_Xsqr = fget_dev(s,Xsqr);
  int m = s->m, N = s->N, n = s->n, k2p2 = s->k2p2;

  dim3 block(8, 8, 1); // Assumes k2p2 <= 8
  dim3 grid(m, 1, 1);
  bfast_step_2<<<grid, block>>>(d_X, d_Xt, d_Y, d_Xsqr, N, n, k2p2);
}

BFAST_BEGIN_TEST(bfast_step_2_test)
  BFAST_BEGIN_INPUTS { BFAST_VALUE_X, BFAST_VALUE_Y } BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_Xsqr } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS
  {
    BFAST_TRANSPOSE(X, transpose),
    BFAST_STEP(bfast_step_2_run)
  }
  BFAST_END_STEPS
BFAST_END_TEST

