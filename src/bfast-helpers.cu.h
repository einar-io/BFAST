#include "bfast.h"
#include <cstdlib>
#include <time.h>
#include <sys/time.h>

#define CUDA_SUCCEED(x) cuda_api_succeed(x, #x, __FILE__, __LINE__)

static inline void cuda_api_succeed(cudaError res, const char *call,
    const char *file, int line)
{
  if (res != cudaSuccess) {
    fprintf(stderr, "%s:%d: CUDA CALL\n  %s\nfailed with error code %d (%s)\n",
        file, line, call, res, cudaGetErrorString(res));
    exit(EXIT_FAILURE);
  }
}

#define IDX_2D(__r,__c,__nc) ((__r) * (__nc) + (__c))
#define CEIL_DIV(a,b) (((a) + (b) - 1) / (b))

static int64_t get_wall_time(void)
{
  struct timeval time;
  assert(gettimeofday(&time, NULL) == 0);
  return time.tv_sec * 1e6 + time.tv_usec;
}

struct timer {
  int64_t start;
  int64_t sum;
  int nruns;
};


static void timer_reset(struct timer *t)
{
  t->sum = t->nruns = t->start = 0;
}

static void timer_start(struct timer *t)
{
  t->start = get_wall_time();
}

static void timer_stop(struct timer *t)
{
  if (t->start != 0) {
    cudaDeviceSynchronize();
    t->sum += (get_wall_time() - t->start);
    t->nruns++;
    t->start = 0;
  }
}

static void timer_report(struct timer *t, const char *what)
{
  if (num_runs > 0) {
    fprintf(stderr, "%14s took %10.2f us (average of %d runs)\n", what,
        t->sum / (float)t->nruns, t->nruns);
  }
}

static void timer_individual_start(struct timer *t, int idx)
{
  if (print_individual) {
    timer_start(&t[idx]);
  }
}

static void timer_individual_stop(struct timer *t, int idx)
{
  if (print_individual) {
    timer_stop(&t[idx]);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Matrix transposition

__global__ void transpose_kernel(float *A, float *B, int heightA, int widthA)
{

  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;

  if(gidx >= widthA || gidy >= heightA) {
    return;
  }

  B[IDX_2D(gidx, gidy, heightA)] = A[IDX_2D(gidy, gidx, widthA)];
}

void transpose(float *d_A, float *d_B, int heightA, int widthA)
{
  dim3 block(16, 16, 1);
  dim3 grid(CEIL_DIV(widthA, block.x),
            CEIL_DIV(heightA, block.y),
            1);
  transpose_kernel<<<grid, block>>>(d_A, d_B, heightA, widthA);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// For bfast_5, bfast_6, bfast_7a

template <class T>
__device__ inline T scaninc_warp_add(volatile T *in)
{
  const unsigned int idx  = threadIdx.x;
  const unsigned int lane = idx & 31;

  // no synchronization needed inside a WARP,
  //   i.e., SIMD execution
  if (lane >= 1)  { in[idx] = in[idx-1]  + in[idx]; }
  if (lane >= 2)  { in[idx] = in[idx-2]  + in[idx]; }
  if (lane >= 4)  { in[idx] = in[idx-4]  + in[idx]; }
  if (lane >= 8)  { in[idx] = in[idx-8]  + in[idx]; }
  if (lane >= 16) { in[idx] = in[idx-16] + in[idx]; }

  return in[idx];
}

template <class T>
__device__ inline void scaninc_block_add(volatile T *in)
{
  const unsigned int idx    = threadIdx.x;
  const unsigned int lane   = idx &  31;
  const unsigned int warpid = idx >> 5;

  T val = scaninc_warp_add(in);
  __syncthreads();

  if (lane == 31) { in[warpid] = val; }
  __syncthreads();

  if (warpid == 0) scaninc_warp_add(in);
  __syncthreads();

  if (warpid > 0) {
    val = in[warpid-1] + val;
  }

  __syncthreads();
  in[idx] = val;
  __syncthreads();
}

