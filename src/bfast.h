#ifndef BFAST_H
#define BFAST_H
#include <inttypes.h>

struct bfast_in {
  int32_t k;
  int32_t n;
  float freq;
  float hfrac;
  float lam;
  float *images;
  int64_t shp[2];
};

struct bfast_out {
  float *breakss;
  int64_t shp[2];
};


#endif // BFAST_H
