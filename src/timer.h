#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

struct timer {
  int64_t start;
  int64_t sum;
  int num_runs;
};

struct statistics {
  double sample_mean;
  double variance;
  double stddev;
  double rds;
};

static int64_t get_wall_time(void)
{
  struct timeval time;
  assert(gettimeofday(&time, NULL) == 0);
  return time.tv_sec * 1e6 + time.tv_usec;
  
}

static void timer_reset(struct timer *t)
{
  t->sum = t->num_runs = t->start = 0;
}

static void timer_start(struct timer *t)
{
  t->start = get_wall_time();
}

static void timer_stop(struct timer *t)
{
  if (t->start != 0) {
    t->sum += (get_wall_time() - t->start);
    t->num_runs++;
    t->start = 0;
  }
}

static float timer_elapsed(struct timer *t)
{
  assert(t->num_runs > 0);
  return t->sum / (float)t->num_runs;
}

static statistics statistify(timer *t, int num_runs) {

  statistics s;

  s->sample_mean = 0.0;
  s->variance = 0.0;
  s->stddev = 0.0;
  s->rds = 0.0;

    
  /*
  for (int i = 0; i < num_runs; i++) {
    timer_elapsed(&t[i]));
 }
    */



  return s;
}
