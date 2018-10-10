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

// For testing with tests/sanity.fut
void write_sanity_outputs(struct bfast_in *inp)
{
  write_scalar(stdout, 0, &i32_info, &inp->k); printf("\n");
  write_scalar(stdout, 0, &i32_info, &inp->n); printf("\n");
  write_scalar(stdout, 0, &f32_info, &inp->freq); printf("\n");
  write_scalar(stdout, 0, &f32_info, &inp->hfrac); printf("\n");
  write_scalar(stdout, 0, &f32_info, &inp->lam); printf("\n");
  write_scalar(stdout, 0, &i64_info, &inp->shp[0]); printf("\n");
  write_scalar(stdout, 0, &i64_info, &inp->shp[1]); printf("\n");
}

void bfast()
{
  BFAST_ASSERT(0);
  //struct bfast_in input;
  //struct bfast_out output;

  //memset(&input, 0, sizeof(struct bfast_in));
  //memset(&output, 0, sizeof(struct bfast_out));

  //read_bfast_input(&input);
  //bfast_naive(&input, &output);
  //write_outputs(&output);

  //free(input.images);

  //if (output.breaks != NULL) {
  //  free(output.breaks);
  //}
}

void sanity()
{
  // Entry point for testing our test system
  struct bfast_in input;
  memset(&input, 0, sizeof(struct bfast_in));
  read_bfast_input(&input);
  write_sanity_outputs(&input);
}

int run_entry(const char *entry)
{
  struct {
    const char *name;
    void (*f)(void);
  } static const entries[] = {
    { "bfast", bfast },
    { "sanity", sanity }
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
  fprintf(stderr, "panic.h, values.h: "
      "Copyright (c) 2013-2018. DIKU, University of Copenhagen\n");

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
