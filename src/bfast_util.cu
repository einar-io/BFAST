#include "bfast_util.cu.h"
#include "timer.h"

static inline size_t get_val_size(struct bfast_value *val)
{
  size_t mem = 1;
  for (int j = 0; j < val->ndims; j++) {
    mem *= val->shp[j];
  }
  mem *= val->type->size;
  return mem;
}

static void state_write_value(struct bfast_state *s, int v)
{
  BFAST_ASSERT(BFAST_VALUE_ID_VALID(v));
  struct bfast_value *val = &s->vals[v];
  size_t mem = get_val_size(val);

  if (val->host == NULL) {
    val->host = malloc(mem);
  }

  CUDA_SUCCEED(cudaMemcpy(val->host, val->dev, mem, cudaMemcpyDeviceToHost));
  write_array(stdout, 1, val->type, val->host, val->shp, val->ndims);
}

static void state_read_value(struct bfast_state *s, int v)
{
  BFAST_ASSERT(BFAST_VALUE_ID_VALID(v));
  struct bfast_value *val = &s->vals[v];

  BFAST_ASSERT(val->host == NULL);
  BFAST_ASSERT(read_array(val->type, &val->host, val->shp, val->ndims) == 0);
}

static void state_read_consts(struct bfast_state *s)
{
  BFAST_ASSERT(read_scalar(&i32_info, &s->k) == 0);
  BFAST_ASSERT(read_scalar(&i32_info, &s->n) == 0);
  BFAST_ASSERT(read_scalar(&f32_info, &s->freq) == 0);
  BFAST_ASSERT(read_scalar(&f32_info, &s->hfrac) == 0);
  BFAST_ASSERT(read_scalar(&f32_info, &s->lam) == 0);
}

#define SET_VAL_DIMS(x,shp0,shp1,shp2,ndim) do { \
  struct bfast_value *val = &s->vals[x]; \
  val->shp[0] = shp0; \
  val->shp[1] = shp1; \
  val->shp[2] = shp2; \
  BFAST_ASSERT(ndim==val->ndims); \
  } while (0);
#define SET_VAL_DIMS_1D(x,shp0) SET_VAL_DIMS(x,shp0,1,1,1)
#define SET_VAL_DIMS_2D(x,shp0,shp1) SET_VAL_DIMS(x,shp0,shp1,1,2)
#define SET_VAL_DIMS_3D(x,shp0,shp1,shp2) SET_VAL_DIMS(x,shp0,shp1,shp2,3)


static void state_set_value_dims(struct bfast_state *s)
{
  struct bfast_value *val_Y = &s->vals[BFAST_VALUE_Y];
  s->k2p2 = s->k * 2 + 2;
  s->m = val_Y->shp[0];
  s->N = val_Y->shp[1];
  s->o = s->N - s->n;

  int k2p2 = s->k2p2, m = s->m, N = s->N, o = s->o;

  SET_VAL_DIMS_2D(         BFAST_VALUE_Y, m,       N       );
  SET_VAL_DIMS_2D(         BFAST_VALUE_X, k2p2,    N       );
  SET_VAL_DIMS_3D(      BFAST_VALUE_Xsqr, m,    k2p2, k2p2 );
  SET_VAL_DIMS_3D(      BFAST_VALUE_Xinv, m,    k2p2, k2p2 );
  SET_VAL_DIMS_2D(     BFAST_VALUE_beta0, m,    k2p2       );
  SET_VAL_DIMS_2D(      BFAST_VALUE_beta, m,    k2p2       );
  SET_VAL_DIMS_2D(   BFAST_VALUE_y_preds, m,       N       );
  SET_VAL_DIMS_1D(       BFAST_VALUE_Nss, m                );
  SET_VAL_DIMS_2D(  BFAST_VALUE_y_errors, m,       N       );
  SET_VAL_DIMS_2D( BFAST_VALUE_val_indss, m,       N       );
  SET_VAL_DIMS_1D(       BFAST_VALUE_nss, m                );
  SET_VAL_DIMS_1D(    BFAST_VALUE_sigmas, m                );
  SET_VAL_DIMS_1D(   BFAST_VALUE_MO_fsts, m                );
  SET_VAL_DIMS_1D(     BFAST_VALUE_BOUND, o                );
  SET_VAL_DIMS_2D(   BFAST_VALUE_breakss, m,       o       );
}

#define SET_VAL_TYPE_INFO(x,_type,_ndims) do {\
  struct bfast_value *val = &s->vals[x]; \
  val->type = _type; \
  val->ndims = _ndims; \
  } while (0);


static struct bfast_state *state_alloc()
{
  struct bfast_state *s =
    (struct bfast_state *)malloc(sizeof(struct bfast_state));
  memset(s, 0, sizeof(struct bfast_state));

  SET_VAL_TYPE_INFO(         BFAST_VALUE_Y, &f32_info, 2 );
  SET_VAL_TYPE_INFO(         BFAST_VALUE_X, &f32_info, 2 );
  SET_VAL_TYPE_INFO(      BFAST_VALUE_Xsqr, &f32_info, 3 );
  SET_VAL_TYPE_INFO(      BFAST_VALUE_Xinv, &f32_info, 3 );
  SET_VAL_TYPE_INFO(     BFAST_VALUE_beta0, &f32_info, 2 );
  SET_VAL_TYPE_INFO(      BFAST_VALUE_beta, &f32_info, 2 );
  SET_VAL_TYPE_INFO(   BFAST_VALUE_y_preds, &f32_info, 2 );
  SET_VAL_TYPE_INFO(       BFAST_VALUE_Nss, &i32_info, 1 );
  SET_VAL_TYPE_INFO(  BFAST_VALUE_y_errors, &f32_info, 2 );
  SET_VAL_TYPE_INFO( BFAST_VALUE_val_indss, &i32_info, 2 );
  SET_VAL_TYPE_INFO(       BFAST_VALUE_nss, &i32_info, 1 );
  SET_VAL_TYPE_INFO(    BFAST_VALUE_sigmas, &f32_info, 1 );
  SET_VAL_TYPE_INFO(   BFAST_VALUE_MO_fsts, &f32_info, 1 );
  SET_VAL_TYPE_INFO(     BFAST_VALUE_BOUND, &f32_info, 1 );
  SET_VAL_TYPE_INFO(   BFAST_VALUE_breakss, &f32_info, 2 );

  return s;
}

static void state_alloc_device(struct bfast_state *s)
{
  for (int i = 0; i < BFAST_NUM_VALUES; i++) {
    struct bfast_value *val = &s->vals[i];
    size_t mem = get_val_size(val);
    if (val->dev == NULL) {
      CUDA_SUCCEED(cudaMalloc(&val->dev, mem));
    }
  }
}

static struct bfast_state *state_init_from_stdin()
{
  struct bfast_state *s = state_alloc();

  state_read_consts(s);
  state_read_value(s, BFAST_VALUE_Y);
  state_set_value_dims(s);
  state_alloc_device(s);

  struct bfast_value *Y_val = &s->vals[BFAST_VALUE_Y];
  CUDA_SUCCEED(cudaMemcpy(Y_val->dev, Y_val->host, get_val_size(Y_val),
                          cudaMemcpyHostToDevice));

  return s;
}

static struct bfast_state *state_init_from_stdin_test(const int *inputs,
    size_t n_inputs)
{
  struct bfast_state *s = state_alloc();

  const int read_vals[] = {
    BFAST_VALUE_Y, BFAST_VALUE_X, BFAST_VALUE_Xsqr, BFAST_VALUE_Xinv,
    BFAST_VALUE_beta0, BFAST_VALUE_beta, BFAST_VALUE_y_preds, BFAST_VALUE_Nss,
    BFAST_VALUE_y_errors, BFAST_VALUE_val_indss, BFAST_VALUE_nss,
    BFAST_VALUE_sigmas, BFAST_VALUE_MO_fsts, BFAST_VALUE_BOUND
  };

  state_read_consts(s);
  for (int i = 0; i < NUM_ELEMS(read_vals); i++) {
    state_read_value(s, read_vals[i]);

    int is_input = 0;
    for (int j = 0; j < n_inputs; j++) {
      if (inputs[j] == read_vals[i]) {
        is_input = 1;
        break;
      }
    }

    if (!is_input) {
      struct bfast_value *val = &s->vals[read_vals[i]];
      free(val->host);
      val->host = NULL;
    }
  }
  state_set_value_dims(s);
  state_alloc_device(s);

  for (int i = 0; i < n_inputs; i++) {
    BFAST_ASSERT(BFAST_VALUE_ID_VALID(inputs[i]));
    struct bfast_value *val = &s->vals[inputs[i]];
    size_t mem = get_val_size(val);
    CUDA_SUCCEED(cudaMemcpy(val->dev, val->host, mem, cudaMemcpyHostToDevice));
  }

  // TODO: Make assertions about shapes, e.g.,
  //   s->vals[BFAST_VALUE_Y].shp[1] == s->vals[BFAST_VALUE_X].shp[1]

  return s;
}

static void state_free(struct bfast_state *s)
{
  for (int i = 0; i < BFAST_NUM_VALUES; i++) {
    struct bfast_value *val = &s->vals[i];
    if (val->host != NULL)   { free(val->host); }
    if (val->dev != NULL)    { CUDA_SUCCEED(cudaFree(val->dev)); }
    if (val->dev_t != NULL)  { CUDA_SUCCEED(cudaFree(val->dev_t)); }
  }
  free(s);
}

static void state_alloc_transposed(struct bfast_state *s, int v)
{
  BFAST_ASSERT(BFAST_VALUE_ID_VALID(v));
  struct bfast_value *val = &s->vals[v];

  if (val->dev_t == NULL) {
    CUDA_SUCCEED(cudaMalloc(&val->dev_t, get_val_size(val)));
  }
}

static void state_alloc_transposed_from_steps(struct bfast_state *s,
    const struct bfast_step *steps, int num_steps)
{
  // Loop through and allocate the transposed matrix buffers we need
  for (int i = 0; i < num_steps; i++) {
    if (steps[i].mode == EXECUTE_TRANSPOSE_KERNEL
        || steps[i].mode == EXECUTE_UNTRANSPOSE_KERNEL) {
      state_alloc_transposed(s, steps[i].v);
    }
  }
}

static void bfast_run_step(struct bfast_state *s,
    const struct bfast_step *step)
{
  switch(step->mode) {
    case EXECUTE_STEP_KERNEL:
      {
        step->k(s);
        break;
      }
    case EXECUTE_TRANSPOSE_KERNEL:
      {
        BFAST_ASSERT(BFAST_VALUE_ID_VALID(step->v));
        struct bfast_value *val = &s->vals[step->v];
        BFAST_ASSERT(val->ndims == 2);
        BFAST_ASSERT(val->dev != NULL);
        BFAST_ASSERT(val->dev_t != NULL);
        step->t(val->dev, val->dev_t, val->shp[0], val->shp[1]);
        break;
      }
    case EXECUTE_UNTRANSPOSE_KERNEL:
      {
        BFAST_ASSERT(BFAST_VALUE_ID_VALID(step->v));
        struct bfast_value *val = &s->vals[step->v];
        BFAST_ASSERT(val->ndims == 2);
        BFAST_ASSERT(val->dev != NULL);
        BFAST_ASSERT(val->dev_t != NULL);
        step->t(val->dev_t, val->dev, val->shp[1], val->shp[0]);
        break;
      }
    default:
      panic(-1, "invalid bfast step type %d\n", step->mode);
  }
}

void bfast_run(const struct bfast_run_config *cfg, const char *name,
               const struct bfast_step *steps, int num_steps)
{
  struct bfast_state *s = state_init_from_stdin();
  state_alloc_transposed_from_steps(s, steps, num_steps);

  CUDA_SUCCEED(cudaDeviceSynchronize());

  if (cfg->measure_steps) {
    struct timer *t = (struct timer *)malloc(num_steps * sizeof(struct timer));
    for (int i = 0; i < num_steps; i++) {
      timer_reset(&t[i]);
    }
    for (int i = 0; i < num_steps; i++) {
      for (int j = 0; j < cfg->num_runs; j++) {
        timer_start(&t[i]);
        bfast_run_step(s, &steps[i]);
        CUDA_SUCCEED(cudaDeviceSynchronize());
        timer_stop(&t[i]);
      }
    }
    if (cfg->print_runtimes) {
      float tot_time = 0;
      for (int i = 0; i < num_steps; i++) {
        tot_time += timer_elapsed(&t[i]);
      }
      fprintf(stderr,"%s:\n", name);
      for (int i = 0; i < num_steps; i++) {
        fprintf(stderr, "  %20s: %10.2f us (average of %d runs)\n",
                steps[i].desc , timer_elapsed(&t[i]), cfg->num_runs);
      }
      fprintf(stderr,"\n\n  %20s: %10.2f us (average of %d runs)\n",
              "Total runtime", tot_time, cfg->num_runs);
    }
    free(t);
  } else {
    struct timer t;
    timer_reset(&t);
    for (int i = 0; i < cfg->num_runs; i++) {
      timer_start(&t);
      for (int j = 0; j < num_steps; j++) {
        bfast_run_step(s, &steps[j]);
      }
      CUDA_SUCCEED(cudaDeviceSynchronize());
      timer_stop(&t);
    }
    if (cfg->print_runtimes) {
      fprintf(stderr, "%20s took %10.2f us (average of %d runs)\n",
              name, timer_elapsed(&t), cfg->num_runs);
    }
  }

  state_write_value(s, BFAST_VALUE_breakss);
  state_free(s);
}

void bfast_run_test(const struct bfast_run_config *cfg,
                    const int *inputs, int num_inputs,
                    const int *outputs, int num_outputs,
                    const struct bfast_step *steps, int num_steps)
{
  // Ignore cfg for now

  struct bfast_state *s = state_init_from_stdin_test(inputs, num_inputs);
  state_alloc_transposed_from_steps(s, steps, num_steps);

  for (int i = 0; i < num_steps; i++) {
    bfast_run_step(s, &steps[i]);
  }

  for (int i = 0; i < num_outputs; i++) {
    state_write_value(s, outputs[i]);
  }
  state_free(s);
}


