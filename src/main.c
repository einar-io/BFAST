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
  fprintf(stderr, "m=%d, k2p2=%d\n", m, k2p2); // XXX

  float *Xinv = NULL;
  bfast_step_3_single(Xsqr, &Xinv, k2p2, m);

  int64_t Xinv_shp[3] = { m, k2p2, k2p2 };
  write_array(stdout, 1, &f32_info, Xinv, Xinv_shp, 3);

  free(Xinv);
  free(Xsqr);
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
    { "bfast-3", bfast_3 }
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
