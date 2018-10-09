#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "bfast.h"

#define CUDA_SUCCEED(x) (assert((x) == cudaSuccess))

#define IDX_2D(__r,__c,__nc) ((__r) * (__nc) + (__c))

__global__ void mk_X(float *X, int32_t k2p2, int32_t N, float f)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= k2p2 || j >= N) {
    return;
  }

  float val;
  if (i == 0) { val = 1.0; }
  else if (i == 1) { val = (float)j; }
  else {
    float angle = 2.0 * pi * (float)(i/2) * (float)j / f;
    if (i % 2 == 0) {
      val = sin(angle);
    } else {
      val = cos(angle);
    }
  }

  X[IDX_2D(i, j, N)] = val;
}

__global__ void mat_transpose(float *A, float *B, int heightA, int widthA) {

  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;

  if((gidx >= widthA) || (gidy >= heightA)) {
    return;
  }

  B[gidx*heightA+gidy] = A[gidy*widthA + gidx];
}

__global__ void block_matmat_filt(float *Xh, float *Xth, float *Yh,
    int k2p2, int n)
{
  // Xh is k2p2 by n
  // Xth is n by k2p2
  // Yh is m by n


}




extern "C" void bfast_naive(struct bfast_in *in, struct bfast_out  *out)
{
  // Step 1: mk_X, mat_transpose
  // Step 2: block_matmat_filt
  // Step 3: block_mat_inv
  // Step 4: matmat_mul_filt, matmat_mul
  // Step 5: kernel for map body. blocksize=N
  // Step 6: kernel for map body. blocksize=n
  // Step 7: kernel for map body, kernel for map body
  // Step 8: kernel or map body, possibly scan



  out->breaks = NULL;
  out->shp[0] = 0;
  out->shp[1] = 0;

  //

}

__global__ void mat_mult(float* A, float* B, float* C, int heightA,
  int widthB, int widthA)
{
  float accum = 0.0f;

  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;

  if((gidx >= widthB) || (gidy >= heightA)) {
    return;
  }

  for(int k = 0; k < widthA; k ++) {
      accum += A[gidy * widthA + k] * B[k * widthB + gidx];
  }

  C[gidy * widthB + gidx] = accum;
}



