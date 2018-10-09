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

#define BFAST_ASSERT(x) do { \
  if (!x) { \
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

// For testing with tests/sanity.fut
void write_sanity_outputs(struct bfast_in *inp)
{
  write_scalar(stdout, 0, &i32_info, &inp->k); printf("\n");
  write_scalar(stdout, 0, &i32_info, &inp->n); printf("\n");
  write_scalar(stdout, 0, &f32_info, &inp->freq); printf("\n");
  write_scalar(stdout, 0, &f32_info, &inp->hfrac); printf("\n");
  write_scalar(stdout, 0, &f32_info, &inp->lam); printf("\n");
  // XXX: Dims are i32 in Futhark, but read_array returns i64's
  write_scalar(stdout, 0, &i32_info, &inp->shp[0]); printf("\n");
  write_scalar(stdout, 0, &i32_info, &inp->shp[1]); printf("\n");
}

int main(int argc, char **argv)
{
  progname = argv[0]; // panic.h
  fprintf(stderr, "panic.h, values.h: "
      "Copyright (c) 2013-2018. DIKU, University of Copenhagen\n");

  struct bfast_in input;
  struct bfast_out output;

  memset(&input, 0, sizeof(struct bfast_in));
  memset(&output, 0, sizeof(struct bfast_out));

  read_bfast_input(&input);
  //bfast_naive(&input, &output);
  //write_outputs(&output);
  write_sanity_outputs(&input);



  free(input.images);

  if (output.breaks != NULL) {
    free(output.breaks);
  }
  return 0;
}
