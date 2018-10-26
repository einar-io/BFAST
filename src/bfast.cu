#include <cstdio>
#include "bfast_util.cu.h"
#include "kernels/bfast_helpers.cu.h"

// bfast_others.cu
void bfast_step_1_run(struct bfast_state *s);
void bfast_step_3_run(struct bfast_state *s);
void bfast_step_4a_run(struct bfast_state *s);
void bfast_step_4b_run(struct bfast_state *s);
void bfast_step_5_run(struct bfast_state *s);
void bfast_step_7a_run(struct bfast_state *s);
void bfast_step_7b_run(struct bfast_state *s);

// bfast_step_2.cu
void bfast_step_2_run(struct bfast_state *s);
void bfast_step_2_tiled_run(struct bfast_state *s);

// bfast_step_4.cu
void bfast_step_4c_run(struct bfast_state *s);
void bfast_step_4c_flipped_run(struct bfast_state *s);

// bfast_step_6.cu
void bfast_step_6_run(struct bfast_state *s);
void bfast_step_6_reuse_run(struct bfast_state *s);

// bfast_step_8.cu
void bfast_step_8_run(struct bfast_state *s);
void bfast_step_8_opt_run(struct bfast_state *s);
void bfast_step_8_opt2_run(struct bfast_state *s);


extern "C" void bfast_naive(struct bfast_run_config *cfg)
{
  const struct bfast_step steps[] = {
    BFAST_STEP(bfast_step_1_run),
    BFAST_TRANSPOSE(X, transpose),
    BFAST_STEP(bfast_step_2_run),
    BFAST_STEP(bfast_step_3_run),
    BFAST_TRANSPOSE(Y, transpose),
    BFAST_STEP(bfast_step_4a_run),
    BFAST_UNTRANSPOSE(beta0, transpose),
    BFAST_STEP(bfast_step_4b_run),
    BFAST_TRANSPOSE(beta, transpose),
    BFAST_STEP(bfast_step_4c_run),
    BFAST_UNTRANSPOSE(y_preds, transpose),
    BFAST_STEP(bfast_step_5_run),
    BFAST_STEP(bfast_step_6_run),
    BFAST_STEP(bfast_step_7a_run),
    BFAST_STEP(bfast_step_7b_run),
    BFAST_STEP(bfast_step_8_run)
  };
  bfast_run(cfg, "bfast-naive", steps, NUM_ELEMS(steps));
}

extern "C" void bfast_opt(struct bfast_run_config *cfg)
{
  const struct bfast_step steps[] = {
    BFAST_STEP(bfast_step_1_run),
    BFAST_TRANSPOSE(X, transpose),
    BFAST_TRANSPOSE(Y, transpose),
    BFAST_STEP(bfast_step_2_tiled_run),
    BFAST_STEP(bfast_step_3_run),
    BFAST_STEP(bfast_step_4a_run),
    BFAST_UNTRANSPOSE(beta0, transpose),
    BFAST_STEP(bfast_step_4b_run),
    BFAST_STEP(bfast_step_4c_flipped_run),
    BFAST_STEP(bfast_step_5_run),
    BFAST_STEP(bfast_step_6_reuse_run),
    BFAST_STEP(bfast_step_7a_run),
    BFAST_STEP(bfast_step_7b_run),
    BFAST_STEP(bfast_step_8_opt2_run)
  };
  bfast_run(cfg, "bfast-opt", steps, NUM_ELEMS(steps));
}


