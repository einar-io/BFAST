#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "bfast.h"

#define CUDA_SUCCEED(x) (assert((x) == cudaSuccess))

__global__ void write_me(float *p)
{
  p[0] = 0.0;
}

extern "C" void bfast_naive(struct bfast_in *in, struct bfast_out  *out)
{
  out->breaks = NULL;
  out->shp[0] = 0;
  out->shp[1] = 0;
}

