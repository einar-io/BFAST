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
#include "../bfast_util.cu.h"
#include "bfast_helpers.cu.h"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Implementation that reuses shared memory

__global__ void bfast_step_6_reuse(float *Yh, float *y_errors, int *nss,
    float *sigmas, int n, int N, int k2p2)
{
  // Grid dimensions (x, y, z): (m, 1, 1)
  // Block dimensions (x, y, z ): (n, 1, 1)

  if (threadIdx.x >= n) { return; }

  float *yh = &Yh[blockIdx.x * N]; // Yh is Y, so N cols in memory
  float *y_error = &y_errors[blockIdx.x * N];

  //__shared__ int num_valids[1024];
  extern __shared__ int num_valids[];
  num_valids[threadIdx.x] = !isnan(yh[threadIdx.x]);
  __syncthreads();
  int val_ns = scaninc_block_add_nowrite<int>(num_valids);
  //int ns = num_valids[n - 1];
  int ns;
  if (threadIdx.x == n-1) {
    ns = val_ns;
  }
  __syncthreads(); // necessary because shared memory is reused

  float *sigma_shared = (float *) num_valids;
  //__shared__ float sigma_shared[1024];
  float val = threadIdx.x < ns ? y_error[threadIdx.x] : 0.0;
  val = val * val;
  sigma_shared[threadIdx.x] = val;
  __syncthreads();
  float val_sigma = scaninc_block_add_nowrite<float>(sigma_shared);

  if (threadIdx.x == n-1) {
    sigmas[blockIdx.x] = __fsqrt_rd(val_sigma / ((float)(ns - k2p2)));
    nss[blockIdx.x] = ns;
  }
  /*
  if (threadIdx.x == 0) {
    sigmas[blockIdx.x] =
      __fsqrt_rd(sigma_shared[n - 1] / ((float)(ns - k2p2)));
    nss[blockIdx.x] = ns;
  }
  */
}

void bfast_step_6_reuse_run(struct bfast_state *s)
{
  int n = s->n, k2p2 = s->k2p2, m = s->m, N = s->N;
  float *d_Y = fget_dev(s,Y), *d_y_errors = fget_dev(s,y_errors);
  int *d_nss = iget_dev(s,nss);
  float *d_sigmas = fget_dev(s,sigmas);

  dim3 block(n, 1, 1);
  dim3 grid(m, 1, 1);
  const size_t shared_mem = n * sizeof(float);
  bfast_step_6_reuse<<<grid, block, shared_mem>>>(d_Y, d_y_errors, d_nss, d_sigmas,
                                      n, N, k2p2);
}

BFAST_BEGIN_TEST(bfast_step_6_reuse_test)
  BFAST_BEGIN_INPUTS { BFAST_VALUE_Y, BFAST_VALUE_y_errors } BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_nss, BFAST_VALUE_sigmas } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS { BFAST_STEP(bfast_step_6_reuse_run) } BFAST_END_STEPS
BFAST_END_TEST

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Naive implementation
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

  __shared__ float sigma_shared[1024];
  float val = threadIdx.x < ns ? y_error[threadIdx.x] : 0.0;
  val = val * val;
  sigma_shared[threadIdx.x] = val;
  __syncthreads();
  scaninc_block_add<float>(sigma_shared);

  if (threadIdx.x == 0) {
    float sigma0 = sigma_shared[n - 1];
    float sigma = sqrtf(sigma0 / ((float)(ns - k2p2)));

    nss[blockIdx.x] = ns;
    sigmas[blockIdx.x] = sigma;
  }
}

void bfast_step_6_run(struct bfast_state *s)
{
  int n = s->n, k2p2 = s->k2p2, m = s->m, N = s->N;
  float *d_Y = fget_dev(s,Y), *d_y_errors = fget_dev(s,y_errors);
  int *d_nss = iget_dev(s,nss);
  float *d_sigmas = fget_dev(s,sigmas);

  dim3 block(n, 1, 1);
  dim3 grid(m, 1, 1);
  bfast_step_6<<<grid, block>>>(d_Y, d_y_errors, d_nss, d_sigmas, n, N, k2p2);
}

BFAST_BEGIN_TEST(bfast_step_6_test)
  BFAST_BEGIN_INPUTS { BFAST_VALUE_Y, BFAST_VALUE_y_errors } BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_nss, BFAST_VALUE_sigmas } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS { BFAST_STEP(bfast_step_6_run) } BFAST_END_STEPS
BFAST_END_TEST
