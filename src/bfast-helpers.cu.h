
#define CUDA_SUCCEED(x) (assert((x) == cudaSuccess))

#define IDX_2D(__r,__c,__nc) ((__r) * (__nc) + (__c))
#define CEIL_DIV(a,b) (((a) + (b) - 1) / (b))

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
// For bfast_5

/*
class TupleInt {
  public:
    int x;
    int y;
    __device__ __host__ inline TupleInt()
    {
        x = 0; y = 0;
    }
    __device__ __host__ inline TupleInt(const int& a, const int& b)
    {
        x = a; y = b;
    }
    __device__ __host__ inline TupleInt(const TupleInt& i2)
    {
        x = i2.x; y = i2.y;
    }
    volatile __device__ __host__ inline TupleInt& operator=(const TupleInt& i2) volatile
    {
        x = i2.x; y = i2.y;
        return *this;
    }
};

TupleInt map_fun(float val)
{
  int nan = isnan(val);
  return TupleInt(!nan, nan);
}

TupleInt op(TupleInt a, TupleInt b)
{
  return TupleInt(a.x + b.x, a.y + b.y);
}

TupleInt scaninc_warp(volatile TupleInt *p)
{
  const unsigned int idx = threadIdx.x
  const unsigned int lane = idx & 31;

  // no synchronization needed inside a WARP,
  //   i.e., SIMD execution
  if (lane >= 1)  p[idx] = op(p[idx-1],  p[idx]);
  if (lane >= 2)  p[idx] = op(p[idx-2],  p[idx]);
  if (lane >= 4)  p[idx] = op(p[idx-4],  p[idx]);
  if (lane >= 8)  p[idx] = op(p[idx-8],  p[idx]);
  if (lane >= 16) p[idx] = op(p[idx-16], p[idx]);

  return const_cast<T&>(p[idx]);
}

TupleInt scaninc_map_warp(volatile float *in, volatile TupleInt *p)
{
  const unsigned int idx = threadIdx.x
  const unsigned int lane = idx & 31;

  // no synchronization needed inside a WARP,
  //   i.e., SIMD execution
  if (lane >= 1)  p[idx] = op(map_fun(in[idx-1]),  map_fun(in[idx]));
  if (lane >= 2)  p[idx] = op(map_fun(in[idx-2]),  map_fun(in[idx]));
  if (lane >= 4)  p[idx] = op(map_fun(in[idx-4]),  map_fun(in[idx]));
  if (lane >= 8)  p[idx] = op(map_fun(in[idx-8]),  map_fun(in[idx]));
  if (lane >= 16) p[idx] = op(map_fun(in[idx-16]), map_fun(in[idx]));

  return const_cast<T&>(p[idx]);
}

void scaninc_map_block(volatile float *in, volatile TupleInt *p)
{
  const unsigned int idx = threadIdx.x
  const unsigned int lane = idx &  31;
  const unsigned int warpid = idx >> 5;

  TupleInt val = scaninc_map_warp<OP>(in, p);
  __syncthreads();

  if (lane == 31) { p[warpid] = val; }
  __syncthreads();

  if (warpid == 0) scaninc_warp(p);
  __syncthreads();

  if (warpid > 0) {
      val = op(p[warpid-1], val);
  }

  __syncthreads();
  p[idx] = val;
}
*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// For bfast_6

/*
float op2(float a, float b) { return a + b; }

float scaninc_warp_op2(volatile float *in, volatile float *p)
{
  const unsigned int idx = threadIdx.x
  const unsigned int lane = idx & 31;

  // no synchronization needed inside a WARP,
  //   i.e., SIMD execution
  if (lane >= 1)  p[idx] = op(in[idx-1],  in[idx]);
  if (lane >= 2)  p[idx] = op(in[idx-2],  in[idx]);
  if (lane >= 4)  p[idx] = op(in[idx-4],  in[idx]);
  if (lane >= 8)  p[idx] = op(in[idx-8],  in[idx]);
  if (lane >= 16) p[idx] = op(in[idx-16], in[idx]);

  return p[idx];
}

void scaninc_block_op2(volatile float *in, volatile float *p)
{
  const unsigned int idx = threadIdx.x
  const unsigned int lane = idx &  31;
  const unsigned int warpid = idx >> 5;

  float val = scaninc_map_warp(in, p);
  __syncthreads();

  if (lane == 31) { p[warpid] = val; }
  __syncthreads();

  if (warpid == 0) { scaninc_warp(p); }
  __syncthreads();

  if (warpid > 0) {
      val = op(p[warpid-1], val);
  }

  __syncthreads();
  p[idx] = val;
}

*/

