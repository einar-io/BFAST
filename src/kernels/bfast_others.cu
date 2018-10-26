#include "../bfast_util.cu.h"
#include "bfast_helpers.cu.h"

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

void bfast_step_1_run(struct bfast_state *s)
{
  float *d_X = fget_dev(s,X);
  int N = s->N, k2p2 = s->k2p2;
  float freq = s->freq;

  dim3 block(16, 16, 1);
  dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(k2p2, block.y), 1);
  bfast_step_1<<<grid, block>>>(d_X, k2p2, N, freq);
}

BFAST_BEGIN_TEST(bfast_step_1_test)
  BFAST_BEGIN_INPUTS { } BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_X } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS { BFAST_STEP(bfast_step_1_run) } BFAST_END_STEPS
BFAST_END_TEST

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

void bfast_step_3_run(struct bfast_state *s)
{
  float *d_Xsqr = fget_dev(s,Xsqr), *d_Xinv = fget_dev(s,Xinv);
  int m = s->m, k2p2 = s->k2p2;

  dim3 block(16, 8, 1); // Assumes k2p2 <= 8
  dim3 grid(m, 1, 1);
  const size_t shared_size = k2p2 * 2 * k2p2 * sizeof(float);
  bfast_step_3<<<grid, block, shared_size>>>(d_Xsqr, d_Xinv, k2p2);
}

BFAST_BEGIN_TEST(bfast_step_3_test)
  BFAST_BEGIN_INPUTS { BFAST_VALUE_Xsqr } BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_Xinv } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS { BFAST_STEP(bfast_step_3_run) } BFAST_END_STEPS
BFAST_END_TEST

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

void bfast_step_4a_run(struct bfast_state *s)
{
  float *d_X = fget_dev(s,X), *d_Y = fget_dev(s,Y), *d_Yt = fget_dev_t(s,Y);
  float *d_beta0t = fget_dev_t(s,beta0);
  int m = s->m, k2p2 = s->k2p2, n = s->n, N = s->N;

  dim3 block(16, 16, 1);
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

void bfast_step_4b_run(struct bfast_state *s)
{
  float *d_Xinv = fget_dev(s,Xinv), *d_beta0 = fget_dev(s,beta0);
  float *d_beta = fget_dev(s,beta);
  int m = s->m, k2p2 = s->k2p2;

  dim3 block(8, 1, 1); // Assumes k2p2 <= 8
  dim3 grid(m, 1, 1);
  bfast_step_4b<<<grid, block>>>(d_Xinv, d_beta0, d_beta, k2p2);
}

BFAST_BEGIN_TEST(bfast_step_4b_test)
  BFAST_BEGIN_INPUTS { BFAST_VALUE_Xinv, BFAST_VALUE_beta0 } BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_beta } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS { BFAST_STEP(bfast_step_4b_run) } BFAST_END_STEPS
BFAST_END_TEST


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

void bfast_step_5_run(struct bfast_state *s)
{
  int m = s->m, N = s->N;
  float *d_Y = fget_dev(s,Y), *d_y_preds = fget_dev(s,y_preds);
  int *d_Nss = iget_dev(s,Nss), *d_val_indss = iget_dev(s,val_indss);
  float *d_y_errors = fget_dev(s,y_errors);

  dim3 block(N, 1, 1);
  dim3 grid(m, 1, 1);
  bfast_step_5<<<grid, block>>>(d_Y, d_y_preds, d_Nss, d_y_errors, d_val_indss, N);
}

BFAST_BEGIN_TEST(bfast_step_5_test)
  BFAST_BEGIN_INPUTS { BFAST_VALUE_Y, BFAST_VALUE_y_preds } BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS
  {
    BFAST_VALUE_Nss,
    BFAST_VALUE_y_errors,
    BFAST_VALUE_val_indss,
  }
  BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS { BFAST_STEP(bfast_step_5_run) } BFAST_END_STEPS
BFAST_END_TEST

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

void bfast_step_7a_run(struct bfast_state *s)
{
  int h = (int)((float)s->n * s->hfrac), N = s->N, m = s->m;
  float *d_y_errors = fget_dev(s,y_errors), *d_MO_fsts = fget_dev(s,MO_fsts);
  int *d_nss = iget_dev(s,nss);

  dim3 grid(m, 1, 1);
  dim3 block(h, 1, 1);
  bfast_step_7a<<<grid, block>>>(d_y_errors, d_nss, h, N, d_MO_fsts);
}

BFAST_BEGIN_TEST(bfast_step_7a_test)
  BFAST_BEGIN_INPUTS
  {
    BFAST_VALUE_y_errors, BFAST_VALUE_nss
  }
  BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_MO_fsts } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS { BFAST_STEP(bfast_step_7a_run) } BFAST_END_STEPS
BFAST_END_TEST

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


    //BOUND[threadIdx.x] = lam * ( frac>__expf(1.0f) ? __fsqrt_rd(__logf(frac)) : 1);
    BOUND[threadIdx.x] = lam * ( frac>expf(1.0f) ? sqrtf(logf(frac)) : 1);

  }
}

void bfast_step_7b_run(struct bfast_state *s)
{
  float lam = s->lam;
  int n = s->n, N = s->N;
  float *d_BOUND = fget_dev(s,BOUND);

  dim3 grid(1, 1, 1);
  dim3 block(N-n, 1, 1);
  bfast_step_7b<<<grid, block>>>(lam, n, N, d_BOUND);
}

BFAST_BEGIN_TEST(bfast_step_7b_test)
  BFAST_BEGIN_INPUTS { } BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_BOUND } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS { BFAST_STEP(bfast_step_7b_run) } BFAST_END_STEPS
BFAST_END_TEST
