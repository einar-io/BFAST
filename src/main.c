#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "panic.h"
#include "values.h"
#include "bfast.h"

extern void bfast_naive(struct bfast_in *in, struct bfast_out  *out);
extern void bfast_step_1_single(float **X, int k2p2, int N, float f);
extern void bfast_step_2_single(float *X, float *Xt, float *Y, float **Xsqr,
    int N, int n, int k2p2, int m);
extern void bfast_step_3_single(float *Xsqr, float **Xinv, int k2p2, int m);
extern void bfast_step_4a_single(float *X, float *Y, float **beta0, int k2p2,
    int n, int m, int N);
extern void bfast_step_4b_single(float *Xinv, float *beta0, float **beta,
    int m, int k2p2);
extern void bfast_step_4c_single(float *Xt, float *beta, float **y_preds,
    int m, int N, int k2p2);
extern void bfast_step_5_single(float *Y, float *y_preds, int **Nss,
    float **y_errors, int **val_indss, int N, int m);
extern void bfast_step_6_single(float *Y, float *y_errors,  int **nss,
    float **sigmas, int n, int k2p2, int m, int N);

// futhark-test only prints our stderr output if we exit with a non-zero exit
// code. For anything that needs to be printed even if we return 0 (e.g.,
// runtimes), use this.
FILE *out = NULL;

#define BFAST_ASSERT(x) do { \
  if (!(x)) { \
    panic(-1, "%s:%d: Assertion failed:\n\t%s\n", __FILE__, __LINE__, #x); \
  } } while(0);

void read_bfast_input(struct bfast_in *inp)
{
  BFAST_ASSERT(read_scalar(&i32_info, &inp->k) == 0);
  BFAST_ASSERT(read_scalar(&i32_info, &inp->n) == 0);
  BFAST_ASSERT(read_scalar(&f32_info, &inp->freq) == 0);
  BFAST_ASSERT(read_scalar(&f32_info, &inp->hfrac) == 0);
  BFAST_ASSERT(read_scalar(&f32_info, &inp->lam) == 0);
  BFAST_ASSERT(read_array(&f32_info, (void **)&inp->images, inp->shp, 2) == 0);
}

void write_bfast_outputs(struct bfast_out *outp)
{
  write_array(stdout, 1, &f32_info, outp->breaks, outp->shp, 2);
  printf("\n");
}

void sanity()
{
  // Entry point for testing our test system
  struct bfast_in input;
  memset(&input, 0, sizeof(struct bfast_in));

  read_bfast_input(&input);

  write_scalar(stdout, 1, &i32_info, &input.k); printf("\n");
  write_scalar(stdout, 1, &i32_info, &input.n); printf("\n");
  write_scalar(stdout, 1, &f32_info, &input.freq); printf("\n");
  write_scalar(stdout, 1, &f32_info, &input.hfrac); printf("\n");
  write_scalar(stdout, 1, &f32_info, &input.lam); printf("\n");
  write_scalar(stdout, 1, &i64_info, &input.shp[0]); printf("\n");
  write_scalar(stdout, 1, &i64_info, &input.shp[1]); printf("\n");

  free(input.images);
}

void bfast()
{
  panic(-1, "unimplemented\n");
  /*
  struct bfast_in input;
  struct bfast_out output;

  memset(&input, 0, sizeof(struct bfast_in));
  memset(&output, 0, sizeof(struct bfast_out));

  read_bfast_input(&input);
  bfast_naive(&input, &output);
  write_outputs(&output);

  free(input.images);

  if (output.breaks != NULL) {
    free(output.breaks);
  }
  */
}

void bfast_1()
{
  int k2p2, N;
  float f;
  BFAST_ASSERT(read_scalar(&i32_info, &k2p2) == 0);
  BFAST_ASSERT(read_scalar(&i32_info, &N) == 0);
  BFAST_ASSERT(read_scalar(&f32_info, &f) == 0);
  fprintf(out, "k2p2=%d, N=%d, f=%f\n", k2p2, N, f);

  float *X = NULL;
  bfast_step_1_single(&X, k2p2, N, f);

  int64_t shp[2] = { k2p2, N };
  write_array(stdout, 1, &f32_info, X, shp, 2);

  free(X);
}

void bfast_2()
{
  int m, N, k2p2, n;
  float *X = NULL, *Xt = NULL, *Y = NULL;
  int64_t X_shp[2], Xt_shp[2], Y_shp[2];
  BFAST_ASSERT(read_array(&f32_info,  (void **)&X, X_shp,  2) == 0);
  BFAST_ASSERT(read_array(&f32_info, (void **)&Xt, Xt_shp, 2) == 0);
  BFAST_ASSERT(read_array(&f32_info,  (void **)&Y, Y_shp,  2) == 0);
  BFAST_ASSERT(read_scalar(&i32_info, &n) == 0);
  BFAST_ASSERT(X_shp[0] == Xt_shp[1]); // k2p2
  BFAST_ASSERT(X_shp[1] == Xt_shp[0] && X_shp[1] == Y_shp[1]); // N
  N = X_shp[1];
  m = Y_shp[0];
  k2p2 = X_shp[0];
  fprintf(out, "m=%d, N=%d, k2p2=%d, n=%d\n", m, N, k2p2, n);

  float *Xsqr = NULL;
  bfast_step_2_single(X, Xt, Y, &Xsqr, N, n, k2p2, m);

  int64_t Xsqr_shp[3] = { m, k2p2, k2p2 };
  write_array(stdout, 1, &f32_info, Xsqr, Xsqr_shp, 3);

  free(X);
  free(Xt);
  free(Y);
  free(Xsqr);
}

void bfast_3()
{
  int m, k2p2;
  float *Xsqr = NULL;
  int64_t Xsqr_shp[3];
  BFAST_ASSERT(read_array(&f32_info, (void **)&Xsqr, Xsqr_shp, 3) == 0);
  BFAST_ASSERT(Xsqr_shp[1] == Xsqr_shp[2]); // k2p2
  m = Xsqr_shp[0];
  k2p2 = Xsqr_shp[1];
  fprintf(out, "m=%d, k2p2=%d\n", m, k2p2);

  float *Xinv = NULL;
  bfast_step_3_single(Xsqr, &Xinv, k2p2, m);

  int64_t Xinv_shp[3] = { m, k2p2, k2p2 };
  write_array(stdout, 1, &f32_info, Xinv, Xinv_shp, 3);

  free(Xinv);
  free(Xsqr);
}

void bfast_4a()
{
  int n, k2p2, N, m;
  float *X = NULL, *Y = NULL;
  int64_t X_shp[2], Y_shp[2];
  BFAST_ASSERT(read_array(&f32_info, (void **)&X, X_shp,  2) == 0);
  BFAST_ASSERT(read_array(&f32_info, (void **)&Y, Y_shp,  2) == 0);
  BFAST_ASSERT(read_scalar(&i32_info, &n) == 0);
  BFAST_ASSERT(X_shp[1] == Y_shp[1]); // N
  k2p2 = X_shp[0];
  N = X_shp[1];
  m = Y_shp[0];
  fprintf(out, "n=%d, k2p2=%d, N=%d, m=%d\n", n, k2p2, N, m);

  float *beta0 = NULL;
  bfast_step_4a_single(X, Y, &beta0, k2p2, n, m, N);

  int64_t beta0_shp[2] = { m, k2p2 };
  write_array(stdout, 1, &f32_info, beta0, beta0_shp, 2);

  free(X);
  free(Y);
  free(beta0);
}

void bfast_4b()
{
  int m, k2p2;
  float *Xinv = NULL, *beta0 = NULL;
  int64_t Xinv_shp[3], beta0_shp[2];
  BFAST_ASSERT(read_array(&f32_info, (void **)&Xinv, Xinv_shp, 3) == 0);
  BFAST_ASSERT(read_array(&f32_info, (void **)&beta0, beta0_shp, 2) == 0);
  BFAST_ASSERT(Xinv_shp[1] == Xinv_shp[2] && Xinv_shp[1] == beta0_shp[1]);
  BFAST_ASSERT(Xinv_shp[0] == beta0_shp[0]); // m
  m = Xinv_shp[0];
  k2p2 = Xinv_shp[1];
  fprintf(out, "m=%d, k2p2=%d\n", m, k2p2);

  float *beta = NULL;
  bfast_step_4b_single(Xinv, beta0, &beta, m, k2p2);

  int64_t beta_shp[2] = { m, k2p2 };
  write_array(stdout, 1, &f32_info, beta, beta_shp, 2);

  free(Xinv);
  free(beta0);
  free(beta);
}

void bfast_4c()
{
  int m, N, k2p2;
  float *Xt = NULL, *beta = NULL;
  int64_t Xt_shp[2], beta_shp[2];
  BFAST_ASSERT(read_array(&f32_info, (void **)&Xt, Xt_shp, 2) == 0);
  BFAST_ASSERT(read_array(&f32_info, (void **)&beta, beta_shp, 2) == 0);
  BFAST_ASSERT(Xt_shp[1] == beta_shp[1]); // k2p2
  N = Xt_shp[0];
  k2p2 = Xt_shp[1];
  m = beta_shp[0];
  fprintf(out, "N=%d, k2p2=%d, m=%d\n", N, k2p2, m);

  float *y_preds = NULL;
  bfast_step_4c_single(Xt, beta, &y_preds, m, N, k2p2);

  int64_t y_preds_shp[2] = { m, N };
  write_array(stdout, 1, &f32_info, y_preds, y_preds_shp, 2);

  free(Xt);
  free(beta);
  free(y_preds);
}

void bfast_5()
{
  int m, N;
  float *Y = NULL, *y_preds = NULL;
  int64_t Y_shp[2], y_preds_shp[2];
  BFAST_ASSERT(read_array(&f32_info, (void **)&Y, Y_shp, 2) == 0);
  BFAST_ASSERT(read_array(&f32_info, (void **)&y_preds, y_preds_shp, 2) == 0);
  BFAST_ASSERT(Y_shp[0] == y_preds_shp[0]); //  m
  BFAST_ASSERT(Y_shp[1] == y_preds_shp[1]); //  N
  m = Y_shp[0];
  N = Y_shp[1];
  fprintf(out, "m=%d, N=%d\n", m, N);

  int *Nss = NULL, *val_indss = NULL;
  float *y_errors = NULL;
  bfast_step_5_single(Y, y_preds, &Nss, &y_errors, &val_indss, N, m);

  int64_t Nss_shp[1] = { m };
  int64_t val_indss_shp[2] = { m, N }, y_errors_shp[2] = { m, N };
  write_array(stdout, 1, &i32_info, Nss, Nss_shp, 1);
  write_array(stdout, 1, &f32_info, y_errors, y_errors_shp, 2);
  write_array(stdout, 1, &i32_info, val_indss, val_indss_shp, 2);

  free(Y);
  free(y_preds);
  free(Nss);
  free(y_errors);
  free(val_indss);
}

void bfast_6()
{
  int m, N, n, k2p2;
  float *Y = NULL, *y_errors = NULL;
  int64_t Y_shp[2], y_errors_shp[2];
  BFAST_ASSERT(read_array(&f32_info, (void **)&Y, Y_shp, 2) == 0);
  BFAST_ASSERT(read_array(&f32_info, (void **)&y_errors, y_errors_shp, 2) ==0);
  BFAST_ASSERT(read_scalar(&i32_info, &n) == 0);
  BFAST_ASSERT(read_scalar(&i32_info, &k2p2) == 0);
  BFAST_ASSERT(Y_shp[0] == y_errors_shp[0]); //  m
  BFAST_ASSERT(Y_shp[1] == y_errors_shp[1]); //  N
  m = Y_shp[0];
  N = Y_shp[1];

  int *nss = NULL;
  float *sigmas = NULL;
  bfast_step_6_single(Y, y_errors, &nss, &sigmas, n, k2p2, m, N);

  int64_t nss_shp[1] = { m }, sigmas_shp[1] = { m };
  write_array(stdout, 1, &i32_info, nss, nss_shp, 1);
  write_array(stdout, 1, &f32_info, sigmas, sigmas_shp, 1);

  free(Y);
  free(y_errors);
  free(nss);
  free(sigmas);
}

int run_entry(const char *entry)
{
  struct {
    const char *name;
    void (*f)(void);
  } static const entries[] = {
    { "sanity", sanity },
    { "bfast", bfast },
    { "bfast-1", bfast_1 },
    { "bfast-2", bfast_2 },
    { "bfast-3", bfast_3 },
    { "bfast-4a", bfast_4a },
    { "bfast-4b", bfast_4b },
    { "bfast-4c", bfast_4c },
    { "bfast-5", bfast_5 },
    { "bfast-6", bfast_6 }
  };

  for (size_t i = 0; i < sizeof(entries)/sizeof(entries[0]); i++) {
    if (strcmp(entry, entries[i].name) == 0) {
      entries[i].f();
      return 0;
    }
  }

  return 1;
}

int main(int argc, const char **argv)
{
  //fprintf(stderr, "panic.h, values.h: "
  //    "Copyright (c) 2013-2018. DIKU, University of Copenhagen\n");

  const char *entry = NULL;
  const char *out_file = NULL;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-e") == 0 && i < argc - 1) {
      entry = argv[i + 1];
      i++; continue;
    }
    if (strcmp(argv[i], "-o") == 0 && i < argc - 1) {
      out_file = argv[i + 1];
      i++; continue;
    }
  }

  if (entry == NULL) {
    fprintf(stderr, "Usage: %s -e ENTRY [-o OUTFILE]\n", argv[0]);
    fprintf(stderr, "  OUTFILE defaults to OUTPUT");
    return 1;
  }

  if (out_file == NULL) {
    out_file = "OUTPUT";
  }

  out = fopen(out_file, "w");
  BFAST_ASSERT(out != NULL);

  int res = run_entry(entry);
  if (res) {
    fprintf(stderr, "No such entry \"%s\"\n", entry);
  }

  fclose(out);
  return res;
}
