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

