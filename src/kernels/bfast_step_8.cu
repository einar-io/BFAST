
////////////////////////////////////////////////////////////////////////////////
//  Step 8: Calculating breakss
//
// Input:
//   y_errors:  [m][N]f32
//   val_indss: [m][N]i32
//   Nss:       [m]i32
//   nss:       [m]i32
//   sigmas:    [m]f32
//   MO_fsts:   [m]f32
//   BOUND:     [N-n]f32
// Output:
//   breakss:   [m][N-n]f32
#include "../bfast_util.cu.h"
#include "bfast_helpers.cu.h"

__global__ void bfast_step_8_simplified(float *y_errors,  // [m][N]
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
  // Grid:  (m, 1, 1)
  // Block: (N-n, 1, 1)

  if (threadIdx.x >= N-n) { return; }

  int   Ns        = Nss       [blockIdx.x];
  int   ns        = nss       [blockIdx.x];
  float sigma     = sigmas    [blockIdx.x];
  float MO_fst    = MO_fsts   [blockIdx.x];
  float *y_error  = &y_errors [blockIdx.x * N];
  int   *val_inds = &val_indss[blockIdx.x * N];
  float *breaks   = &breakss  [blockIdx.x * (N-n)];
  float val;
  extern __shared__ float MO_shr[];

  if      ( Ns-ns       <= threadIdx.x ) { MO_shr[threadIdx.x] = 0.0f;   }
  else if ( threadIdx.x == 0           ) { MO_shr[threadIdx.x] = MO_fst; }
  else {
    MO_shr[threadIdx.x] =
      -y_error[ns - h + threadIdx.x]+ y_error[ns + threadIdx.x];
  }
  __syncthreads();
  val = scaninc_block_add_nowrite<float>(MO_shr);
  __syncthreads();

  MO_shr[threadIdx.x] = NAN; // overwrite *every* element
  __syncthreads();
  if (threadIdx.x < Ns - ns) {
    val /= (sigma * sqrtf((float)ns));
    MO_shr[val_inds[threadIdx.x + ns] - n] = val;
  }
  __syncthreads();

  float m = MO_shr[threadIdx.x];
  float b = BOUND [threadIdx.x];

  // breaks
  if (isnan(m) || isnan(b)) { breaks[threadIdx.x] = 0.0f; }
  else                      { breaks[threadIdx.x] = fabsf(m) - b; }

}

void bfast_step_8_simplified_run(struct bfast_state *s)
{
  float *d_y_errors = fget_dev(s,y_errors);
  int *d_val_indss = iget_dev(s,val_indss);
  int *d_Nss = iget_dev(s,Nss), *d_nss = iget_dev(s,nss);
  float *d_sigmas = fget_dev(s,sigmas), *d_MO_fsts = fget_dev(s,MO_fsts);
  float *d_BOUND = fget_dev(s,BOUND), *d_breakss = fget_dev(s,breakss);
  int h = (int)((float)s->n * s->hfrac), m = s->m;
  int N = s->N, n = s->n;

  dim3 grid(m, 1, 1);
  dim3 block(N-n, 1, 1);
  const size_t shared_size = (N-n) * sizeof(float);
  bfast_step_8_simplified<<<grid, block, shared_size>>>(d_y_errors, d_val_indss,
                                                  d_Nss, d_nss, d_sigmas,
                                                  d_MO_fsts, d_BOUND, h, n,
                                                  N, d_breakss);
}

BFAST_BEGIN_TEST(bfast_step_8_simplified_test)
  BFAST_BEGIN_INPUTS
  {
    BFAST_VALUE_y_errors, BFAST_VALUE_val_indss, BFAST_VALUE_Nss,
    BFAST_VALUE_nss, BFAST_VALUE_sigmas, BFAST_VALUE_MO_fsts, BFAST_VALUE_BOUND
  }
  BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_breakss } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS { BFAST_STEP(bfast_step_8_simplified_run) } BFAST_END_STEPS
BFAST_END_TEST


__global__ void bfast_step_8_opt2(float *y_errors,  // [m][N]
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
  // Grid:  (m, 1, 1)
  // Block: (N-n, 1, 1)

  if (threadIdx.x >= N-n) { return; }

  int   Ns        = Nss       [blockIdx.x];
  int   ns        = nss       [blockIdx.x];
  float sigma     = sigmas    [blockIdx.x];
  float MO_fst    = MO_fsts   [blockIdx.x];
  float *y_error  = &y_errors [blockIdx.x * N];
  int   *val_inds = &val_indss[blockIdx.x * N];
  float *breaks   = &breakss  [blockIdx.x * (N-n)];
  float val;
  extern __shared__ float shr[];
  volatile float *BOUND_shr = &shr[0];
  volatile float *MO_shr = &shr[N-n];

  BOUND_shr[threadIdx.x] = BOUND[threadIdx.x];

  if      ( Ns-ns       <= threadIdx.x ) { MO_shr[threadIdx.x] = 0.0f;   }
  else if ( threadIdx.x == 0           ) { MO_shr[threadIdx.x] = MO_fst; }
  else {
    MO_shr[threadIdx.x] =
      -y_error[ns - h + threadIdx.x]+ y_error[ns + threadIdx.x];
  }
  __syncthreads();
  val = scaninc_block_add_nowrite<float>(MO_shr);
  __syncthreads();

  MO_shr[threadIdx.x] = NAN; // overwrite *every* element
  __syncthreads();
  if (threadIdx.x < Ns - ns) {
    val /= (sigma * sqrtf((float)ns));
    MO_shr[val_inds[threadIdx.x + ns] - n] = val;
  }
  __syncthreads();

  float m = MO_shr   [threadIdx.x];
  float b = BOUND_shr[threadIdx.x];

  // breaks
  if (isnan(m) || isnan(b)) { breaks[threadIdx.x] = 0.0f; }
  else                      { breaks[threadIdx.x] = fabsf(m) - b; }

}

void bfast_step_8_opt2_run(struct bfast_state *s)
{
  float *d_y_errors = fget_dev(s,y_errors);
  int *d_val_indss = iget_dev(s,val_indss);
  int *d_Nss = iget_dev(s,Nss), *d_nss = iget_dev(s,nss);
  float *d_sigmas = fget_dev(s,sigmas), *d_MO_fsts = fget_dev(s,MO_fsts);
  float *d_BOUND = fget_dev(s,BOUND), *d_breakss = fget_dev(s,breakss);
  int h = (int)((float)s->n * s->hfrac), m = s->m;
  int N = s->N, n = s->n;

  dim3 grid(m, 1, 1);
  dim3 block(N-n, 1, 1);
  const size_t shared_size = (N-n) * 2 * sizeof(float);
  bfast_step_8_opt2<<<grid, block, shared_size>>>(d_y_errors, d_val_indss,
                                                  d_Nss, d_nss, d_sigmas,
                                                  d_MO_fsts, d_BOUND, h, n,
                                                  N, d_breakss);
}

BFAST_BEGIN_TEST(bfast_step_8_opt2_test)
  BFAST_BEGIN_INPUTS
  {
    BFAST_VALUE_y_errors, BFAST_VALUE_val_indss, BFAST_VALUE_Nss,
    BFAST_VALUE_nss, BFAST_VALUE_sigmas, BFAST_VALUE_MO_fsts, BFAST_VALUE_BOUND
  }
  BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_breakss } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS { BFAST_STEP(bfast_step_8_opt2_run) } BFAST_END_STEPS
BFAST_END_TEST

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Optimized (XXX: how?) implementation
//
// XXX: Concisely describe how it is optimized, possibly give the kernel a
// better name. see bfast_step_2/bfast_step_6/bfast_step_4c

__global__ void bfast_step_8_opt(float *y_errors,  // [m][N]
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


/*
  __shared__ float BOUND_shr[1024]; 
  if (threadIdx.x < N) {
    BOUND_shr[threadIdx.x] = BOUND[threadIdx.x];
  }


  //__shared__ int val_inds_shr[1024];

  //if (threadIdx.x < N) {
  //  val_inds_shr[threadIdx.x] = val_inds[threadIdx.x];
  //}
  */

  //__shared__ float MO_shr[1024];
  extern __shared__ float MO_shr[];
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
    //MO_shr[threadIdx.x] /= sigma * sqrtf( (float)ns );
  }

  {
    __syncthreads();

    if ( threadIdx.x < Ns - ns ) {
      val = MO_shr[threadIdx.x];
    }
    else {
      val = NAN;
    }

    // Make sure all threads have read into `val` before overwriting source.
    __syncthreads();
    MO_shr[val_inds[threadIdx.x + ns] - n] = val;
    //int idx = val_inds[threadIdx.x + ns] - n;
    //if ( 0 <= idx && idx < 1024 ){
    //  MO_shr[idx] = val;
    //}
  }

  // Here is a producer/consumer dependency in MO_shr.

  {
    // breaks = ..
    __syncthreads();
    float m = MO_shr   [threadIdx.x];
    float b = BOUND[threadIdx.x];

    if (isnan(m) || isnan(b)) { breaks[threadIdx.x] = 0.0f; }
    else                      { breaks[threadIdx.x] = fabsf(m) - b; }
  }

}

void bfast_step_8_opt_run(struct bfast_state *s)
{
  float *d_y_errors = fget_dev(s,y_errors);
  int *d_val_indss = iget_dev(s,val_indss);
  int *d_Nss = iget_dev(s,Nss), *d_nss = iget_dev(s,nss);
  float *d_sigmas = fget_dev(s,sigmas), *d_MO_fsts = fget_dev(s,MO_fsts);
  float *d_BOUND = fget_dev(s,BOUND), *d_breakss = fget_dev(s,breakss);
  int h = (int)((float)s->n * s->hfrac), m = s->m;
  int N = s->N, n = s->n;

  dim3 grid(m, 1, 1);
  dim3 block(N-n, 1, 1);
  size_t shared_mem = (N-n) * sizeof(float);
  bfast_step_8_opt<<<grid, block, shared_mem>>>(d_y_errors, d_val_indss, d_Nss, d_nss,
                                    d_sigmas, d_MO_fsts, d_BOUND, h, n, N,
                                    d_breakss);
}

BFAST_BEGIN_TEST(bfast_step_8_opt_test)
  BFAST_BEGIN_INPUTS
  {
    BFAST_VALUE_y_errors, BFAST_VALUE_val_indss, BFAST_VALUE_Nss,
    BFAST_VALUE_nss, BFAST_VALUE_sigmas, BFAST_VALUE_MO_fsts, BFAST_VALUE_BOUND
  }
  BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_breakss } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS { BFAST_STEP(bfast_step_8_opt_run) } BFAST_END_STEPS
BFAST_END_TEST

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Naive implementation

#define INVALID_INDEX (-1)

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
  // Read bound into shared memory
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

  __shared__ float MO_shr[1024];
  {
    unsigned int j = threadIdx.x;
    if      ( Ns-ns <= j ) { MO_shr[j] = 0.0f; }
    else if ( j     == 0 ) { MO_shr[j] = MO_fst; }
    else                   { MO_shr[j] = -y_error[ns - h + j] + y_error[ns + j]; }
    __syncthreads();
    scaninc_block_add<float>(MO_shr);
  }

  {
    // MO'
    __syncthreads();
    MO_shr[threadIdx.x] /= sigma * sqrtf( (float)ns );
  }

  __shared__ int val_indsP[1024];
  {
    // val_inds'
    unsigned int i = threadIdx.x;
    __syncthreads();
      if      ( i < Ns - ns ) { val_indsP[i] = val_inds[i + ns] - n; }
      else                    { val_indsP[i] = INVALID_INDEX; }
  }

  __shared__ float MOPP_shr[1024];
  {
    // MO'' = scatter ..
    // NAN initialize
    __syncthreads();
    MOPP_shr[threadIdx.x] = NAN;
    __syncthreads();

    int k = val_indsP[threadIdx.x];
    if ( !(k == INVALID_INDEX) ) {
      MOPP_shr[k] = MO_shr[threadIdx.x];
    }
  }

  {
    // breaks = ..
    __syncthreads();
    float m = MOPP_shr[threadIdx.x];
    float b = BOUND   [threadIdx.x];

    if (isnan(m) || isnan(b)) { breaks[threadIdx.x] = 0.0f; }
    else                      { breaks[threadIdx.x] = fabsf(m) - b; }
  }
}

void bfast_step_8_run(struct bfast_state *s)
{
  float *d_y_errors = fget_dev(s,y_errors);
  int *d_val_indss = iget_dev(s,val_indss);
  int *d_Nss = iget_dev(s,Nss), *d_nss = iget_dev(s,nss);
  float *d_sigmas = fget_dev(s,sigmas), *d_MO_fsts = fget_dev(s,MO_fsts);
  float *d_BOUND = fget_dev(s,BOUND), *d_breakss = fget_dev(s,breakss);
  int h = (int)((float)s->n * s->hfrac), m = s->m;
  int N = s->N, n = s->n;

  dim3 grid(m, 1, 1);
  dim3 block(1024, 1, 1);
  bfast_step_8<<<grid, block>>>(d_y_errors, d_val_indss, d_Nss, d_nss,
                                d_sigmas, d_MO_fsts, d_BOUND, h, n, N,
                                d_breakss);
}

BFAST_BEGIN_TEST(bfast_step_8_test)
  BFAST_BEGIN_INPUTS
  {
    BFAST_VALUE_y_errors, BFAST_VALUE_val_indss, BFAST_VALUE_Nss,
    BFAST_VALUE_nss, BFAST_VALUE_sigmas, BFAST_VALUE_MO_fsts, BFAST_VALUE_BOUND
  }
  BFAST_END_INPUTS
  BFAST_BEGIN_OUTPUTS { BFAST_VALUE_breakss } BFAST_END_OUTPUTS
  BFAST_BEGIN_STEPS { BFAST_STEP(bfast_step_8_run) } BFAST_END_STEPS
BFAST_END_TEST
