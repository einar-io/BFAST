#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "panic.h"
#include "values.h"
#include "bfast.h"

extern void bfast_step_1_test(struct bfast_run_config *);
extern void bfast_step_2_test(struct bfast_run_config *);
extern void bfast_step_2_tiled_test(struct bfast_run_config *);
extern void bfast_step_3_test(struct bfast_run_config *);
extern void bfast_step_4a_test(struct bfast_run_config *);
extern void bfast_step_4b_test(struct bfast_run_config *);
extern void bfast_step_4c_test(struct bfast_run_config *);
extern void bfast_step_4c_flipped_test(struct bfast_run_config *);
extern void bfast_step_5_test(struct bfast_run_config *);
extern void bfast_step_6_test(struct bfast_run_config *);
extern void bfast_step_6_reuse_test(struct bfast_run_config *);
extern void bfast_step_7a_test(struct bfast_run_config *);
extern void bfast_step_7b_test(struct bfast_run_config *);
extern void bfast_step_8_test(struct bfast_run_config *);
extern void bfast_step_8_opt_test(struct bfast_run_config *);
extern void bfast_naive(struct bfast_run_config *);
extern void bfast_opt(struct bfast_run_config *cfg);

// futhark-test only prints our stderr output if we exit with a non-zero exit
// code. For anything that needs to be printed even if we return 0, use this.
FILE *out = NULL;

int run_entry(const char *entry, struct bfast_run_config *cfg)
{
  struct {
    const char *name;
    void (*f)(struct bfast_run_config *);
  } static const entries[] = {
    // Individual kernel tests
    {          "bfast-1",          bfast_step_1_test },
    {          "bfast-2",          bfast_step_2_test },
    {    "bfast-2-tiled",    bfast_step_2_tiled_test },
    {          "bfast-3",          bfast_step_3_test },
    {         "bfast-4a",         bfast_step_4a_test },
    {         "bfast-4b",         bfast_step_4b_test },
    {         "bfast-4c",         bfast_step_4c_test },
    { "bfast-4c-flipped", bfast_step_4c_flipped_test },
    {          "bfast-5",          bfast_step_5_test },
    {          "bfast-6",          bfast_step_6_test },
    {    "bfast-6-reuse",    bfast_step_6_reuse_test },
    {         "bfast-7a",         bfast_step_7a_test },
    {         "bfast-7b",         bfast_step_7b_test },
    {          "bfast-8",          bfast_step_8_test },
    {      "bfast-8-opt",      bfast_step_8_opt_test },
    // Full BFAST runs
    {      "bfast-naive",                bfast_naive },
    {        "bfast-opt",                  bfast_opt },
  };

  for (size_t i = 0; i < sizeof(entries)/sizeof(entries[0]); i++) {
    if (strcmp(entry, entries[i].name) == 0) {
      entries[i].f(cfg);
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

  struct bfast_run_config cfg;
  cfg.num_runs = 1;
  cfg.measure_steps = 0;
  cfg.print_runtimes = 0;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-e") == 0 && i < argc - 1) {
      entry = argv[i + 1];
      i++; continue;
    } else if (strcmp(argv[i], "-o") == 0 && i < argc - 1) {
      out_file = argv[i + 1];
      i++; continue;
    } else if (strcmp(argv[i], "-r") == 0 && i < argc - 1) {
      cfg.print_runtimes = 1;
      cfg.num_runs = atoi(argv[i + 1]);
      if (cfg.num_runs <= 0) {
        fprintf(stderr, "Number of runs must be positive, not %n\n",
                cfg.num_runs);
        return 1;
      }
      i++; continue;
    } else if (strcmp(argv[i], "-i") == 0 && i < argc - 1) {
      cfg.print_runtimes = 1;
      if (strcmp(argv[i + 1], "0") == 0) {
        cfg.measure_steps = 0;
      } else if (strcmp(argv[i + 1], "1") == 0) {
        cfg.measure_steps = 1;
      } else {
        fprintf(stderr, "Flag must be 0 or 1, not %s\n", argv[i + 1]);
      }
      i++; continue;
    } else {
      // ignore other parameters
    }
  }

  if (entry == NULL) {
    fprintf(stderr, "Usage: %s -e ENTRY [-o OUTFILE]\n", argv[0]);
    fprintf(stderr, "  OUTFILE defaults to OUTPUT\n");
    return 1;
  }

  if (out_file == NULL) {
    out_file = "OUTPUT";
  }

  out = fopen(out_file, "w");
  BFAST_ASSERT(out != NULL);

  int res = run_entry(entry, &cfg);
  if (res) {
    fprintf(stderr, "No such entry \"%s\"\n", entry);
  }

  fclose(out);
  return res;
}
