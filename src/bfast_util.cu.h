#ifndef _BFAST_VALUES_H
#define _BFAST_VALUES_H
#include "panic.h"
#include "values.h"
#include "bfast.h"

#define CUDA_SUCCEED(x) _cuda_api_succeed(x, #x, __FILE__, __LINE__)

static inline void _cuda_api_succeed(cudaError res, const char *call,
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

#define NUM_ELEMS(x) (sizeof(x)/sizeof(x[0]))

enum {
  BFAST_VALUE_Y = 0,
  BFAST_VALUE_X,
  BFAST_VALUE_Xsqr,
  BFAST_VALUE_Xinv,
  BFAST_VALUE_beta0,
  BFAST_VALUE_beta,
  BFAST_VALUE_y_preds,
  BFAST_VALUE_Nss,
  BFAST_VALUE_y_errors,
  BFAST_VALUE_val_indss,
  BFAST_VALUE_nss,
  BFAST_VALUE_sigmas,
  BFAST_VALUE_MO_fsts,
  BFAST_VALUE_BOUND,
  BFAST_VALUE_breakss,
  BFAST_NUM_VALUES
};

#define BFAST_VALUE_ID_VALID(v) ((v) >= 0 && (v) < BFAST_NUM_VALUES)

struct bfast_value {
  void *host;
  void *dev;
  void *dev_t;
  int ndims;
  int64_t shp[3];
  struct primtype_info_t *type;
};

// For when running bfast
struct bfast_state {
  bfast_value vals[BFAST_NUM_VALUES];
  int k;
  int n;
  float freq;
  float hfrac;
  float lam;
  int k2p2;
  int m;
  int N;
  int o; // N - n
};

static inline const char *bfast_get_value_name(int v)
{
  // No designated initializers because C++
  switch (v) {
    case BFAST_VALUE_Y:         return "Y";
    case BFAST_VALUE_X:         return "X";
    case BFAST_VALUE_Xsqr:      return "Xsqr";
    case BFAST_VALUE_Xinv:      return "Xinv";
    case BFAST_VALUE_beta0:     return "beta0";
    case BFAST_VALUE_beta:      return "beta";
    case BFAST_VALUE_y_preds:   return "y_preds";
    case BFAST_VALUE_Nss:       return "Nss";
    case BFAST_VALUE_y_errors:  return "y_errors";
    case BFAST_VALUE_val_indss: return "val_indss";
    case BFAST_VALUE_nss:       return "nss";
    case BFAST_VALUE_sigmas:    return "sigmas";
    case BFAST_VALUE_MO_fsts:   return "MO_fsts";
    case BFAST_VALUE_BOUND:     return "BOUND";
    case BFAST_VALUE_breakss:   return "breakss";
    default:                    return "???";
  }
}

#define bfast_val(v) (BFAST_VALUE_##v)

#define get_host(s, v) _state_safe_get(s, bfast_val(v),  0, __FILE__, __LINE__)
#define get_dev(s, v) _state_safe_get(s, bfast_val(v),  1, __FILE__, __LINE__)
#define get_dev_t(s, v) _state_safe_get(s, bfast_val(v),  2, __FILE__, __LINE__)

#define iget_host(s,v) ((int *)get_host(s,v))
#define fget_host(s,v) ((float *)get_host(s,v))
#define iget_dev(s,v) ((int *)get_dev(s,v))
#define fget_dev(s,v) ((float *)get_dev(s,v))
#define iget_dev_t(s,v) ((int *)get_dev_t(s,v))
#define fget_dev_t(s,v) ((float *)get_dev_t(s,v))

static void _state_get_failed(int v, int which, const char *file, int line)
{
  const char *version;
  switch (which) {
    case 0: version = "host"; break;
    case 1: version = "device"; break;
    case 2: version = "transposed device"; break;
    default: version = "unknown";
  }
  panic(-1, "%s:%d: Tried to get %s version of %s, but it is NULL\n",
      file, line, version, bfast_get_value_name(v));
}

static inline void *_state_safe_get(struct bfast_state *s, int v, int which,
    const char *file, int line)
{
  if (!BFAST_VALUE_ID_VALID(v)) {
    panic(-1, "%s:%d: Tried to get invalid value id %d\n", file, line, v);
  }
  void *p = NULL;
  switch (which) {
    case 0: p = s->vals[v].host; break;
    case 1: p = s->vals[v].dev; break;
    case 2: p = s->vals[v].dev_t; break;
    default: panic(-1, "%s:%d: Should not reach this\n", file, line);
  }
  if (p != NULL) { return p; }
  _state_get_failed(v, which, file, line);
  return NULL; // unreachable
}


// NB!
typedef void (*transpose_kernel_t)(void *, void *, int, int);

typedef void (*bfast_kernel_t)(struct bfast_state *);

struct bfast_step {
  int mode;
  const char *desc;
  bfast_kernel_t k;
  int v;
  transpose_kernel_t t;
};

#define EXECUTE_STEP_KERNEL        0
#define EXECUTE_TRANSPOSE_KERNEL   1
#define EXECUTE_UNTRANSPOSE_KERNEL 2

#define BFAST_STEP(_k) { \
  .mode = EXECUTE_STEP_KERNEL, \
  .desc = #_k, \
  .k = _k, \
  .v = -1, \
  .t = NULL \
}
#define BFAST_TRANSPOSE(_v,_t) { \
  .mode = EXECUTE_TRANSPOSE_KERNEL, \
  .desc = "transpose_"#_v, \
  .k = NULL, \
  .v = bfast_val(_v), \
  .t = _t \
}
#define BFAST_UNTRANSPOSE(_v,_t) { \
  .mode = EXECUTE_UNTRANSPOSE_KERNEL, \
  .desc = "untranspose_"#_v, \
  .k = NULL, \
  .v = bfast_val(_v), \
  .t = _t \
}

#define BFAST_BEGIN_TEST(name) extern "C" void \
  name(const struct bfast_run_config *cfg) {
#define BFAST_END_TEST bfast_run_test(cfg, inputs, NUM_ELEMS(inputs), \
    outputs, NUM_ELEMS(outputs), steps, NUM_ELEMS(steps)); }
#define BFAST_BEGIN_INPUTS const int inputs[] =
#define BFAST_END_INPUTS ;
#define BFAST_BEGIN_OUTPUTS const int outputs[] =
#define BFAST_END_OUTPUTS ;
#define BFAST_BEGIN_STEPS const struct bfast_step steps[] =
#define BFAST_END_STEPS ;


void bfast_run(const struct bfast_run_config *cfg, const char *name,
    const struct bfast_step *steps, int num_steps);
void bfast_run_test(const struct bfast_run_config *cfg, const int *inputs,
    int num_inputs, const int *outputs, int num_outputs,
    const struct bfast_step *steps, int num_steps);


#endif // _BFAST_VALUES_H
