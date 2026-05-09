#define _GNU_SOURCE

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
  kCacheLineBytes = 64,
  kMinWorksetBytes = sizeof(uint64_t),
};

typedef struct {
  int thread_index;
  int cpu;
  uint64_t iters;
  uint64_t line_count;
  _Atomic uint64_t* shared_words;
  pthread_barrier_t* start_barrier;
  pthread_barrier_t* stop_barrier;
  uint64_t touches;
} ThreadCtx;

static void usage(const char* prog) {
  fprintf(stderr,
          "Usage: %s --threads N --cpus c0,c1,... --workset-bytes N --iters N\n",
          prog);
}

static bool parse_u64(const char* text, uint64_t* out) {
  char* end = NULL;
  errno = 0;
  unsigned long long value = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    return false;
  }
  *out = (uint64_t)value;
  return true;
}

static bool parse_i32(const char* text, int* out) {
  char* end = NULL;
  errno = 0;
  long value = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    return false;
  }
  *out = (int)value;
  return true;
}

static int parse_cpu_list(const char* text, int** cpus_out) {
  char* copy = strdup(text);
  if (copy == NULL) {
    return -1;
  }

  int count = 0;
  for (char* p = copy; *p != '\0'; ++p) {
    if (*p == ',') {
      ++count;
    }
  }
  ++count;

  int* cpus = calloc((size_t)count, sizeof(int));
  if (cpus == NULL) {
    free(copy);
    return -1;
  }

  int idx = 0;
  char* saveptr = NULL;
  for (char* token = strtok_r(copy, ",", &saveptr); token != NULL;
       token = strtok_r(NULL, ",", &saveptr)) {
    if (!parse_i32(token, &cpus[idx])) {
      free(cpus);
      free(copy);
      return -1;
    }
    ++idx;
  }

  free(copy);
  *cpus_out = cpus;
  return idx;
}

static double elapsed_sec(const struct timespec* start,
                          const struct timespec* end) {
  const time_t sec = end->tv_sec - start->tv_sec;
  const long nsec = end->tv_nsec - start->tv_nsec;
  return (double)sec + (double)nsec / 1e9;
}

static void* run_writer(void* arg) {
  ThreadCtx* ctx = (ThreadCtx*)arg;

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(ctx->cpu, &cpuset);
  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
    perror("pthread_setaffinity_np");
    return (void*)1;
  }

  pthread_barrier_wait(ctx->start_barrier);

  const uint64_t write_value = (uint64_t)(ctx->thread_index + 1);
  uint64_t touches = 0;
  for (uint64_t iter = 0; iter < ctx->iters; ++iter) {
    for (uint64_t line = 0; line < ctx->line_count; ++line) {
      atomic_store_explicit(
          &ctx->shared_words[line * (kCacheLineBytes / sizeof(uint64_t))],
          write_value, memory_order_relaxed);
      ++touches;
    }
  }

  ctx->touches = touches;

  pthread_barrier_wait(ctx->stop_barrier);
  return NULL;
}

int main(int argc, char** argv) {
  int threads = -1;
  int* cpus = NULL;
  int cpu_count = -1;
  uint64_t workset_bytes = 0;
  uint64_t iters = 0;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      if (!parse_i32(argv[++i], &threads)) {
        fprintf(stderr, "invalid threads value\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--cpus") == 0 && i + 1 < argc) {
      cpu_count = parse_cpu_list(argv[++i], &cpus);
      if (cpu_count <= 0) {
        fprintf(stderr, "invalid cpus list\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--workset-bytes") == 0 && i + 1 < argc) {
      if (!parse_u64(argv[++i], &workset_bytes)) {
        fprintf(stderr, "invalid workset-bytes value\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      if (!parse_u64(argv[++i], &iters)) {
        fprintf(stderr, "invalid iters value\n");
        return 1;
      }
    } else {
      usage(argv[0]);
      return 1;
    }
  }

  if (threads <= 0 || cpu_count <= 0 || workset_bytes == 0 || iters == 0) {
    usage(argv[0]);
    return 1;
  }
  if (cpu_count != threads) {
    fprintf(stderr, "cpus count must match threads\n");
    free(cpus);
    return 1;
  }
  if (workset_bytes < kMinWorksetBytes) {
    fprintf(stderr, "workset-bytes must be at least %d\n", kMinWorksetBytes);
    free(cpus);
    return 1;
  }

  const uint64_t line_count =
      (workset_bytes + kCacheLineBytes - 1) / kCacheLineBytes;
  const size_t alloc_bytes = (size_t)line_count * kCacheLineBytes;

  _Atomic uint64_t* shared_words = NULL;
  if (posix_memalign((void**)&shared_words, kCacheLineBytes, alloc_bytes) != 0) {
    fprintf(stderr, "posix_memalign failed\n");
    free(cpus);
    return 1;
  }

  for (uint64_t line = 0; line < line_count; ++line) {
    atomic_init(
        &shared_words[line * (kCacheLineBytes / sizeof(uint64_t))], 0);
  }

  pthread_barrier_t start_barrier;
  pthread_barrier_t stop_barrier;
  pthread_barrier_init(&start_barrier, NULL, (unsigned)threads + 1);
  pthread_barrier_init(&stop_barrier, NULL, (unsigned)threads + 1);

  pthread_t* thread_handles = calloc((size_t)threads, sizeof(pthread_t));
  ThreadCtx* thread_ctx = calloc((size_t)threads, sizeof(ThreadCtx));
  if (thread_handles == NULL || thread_ctx == NULL) {
    fprintf(stderr, "allocation failed\n");
    free(thread_handles);
    free(thread_ctx);
    free(shared_words);
    free(cpus);
    return 1;
  }

  for (int i = 0; i < threads; ++i) {
    thread_ctx[i].thread_index = i;
    thread_ctx[i].cpu = cpus[i];
    thread_ctx[i].iters = iters;
    thread_ctx[i].line_count = line_count;
    thread_ctx[i].shared_words = shared_words;
    thread_ctx[i].start_barrier = &start_barrier;
    thread_ctx[i].stop_barrier = &stop_barrier;
    if (pthread_create(&thread_handles[i], NULL, run_writer, &thread_ctx[i]) !=
        0) {
      fprintf(stderr, "pthread_create failed\n");
      free(thread_handles);
      free(thread_ctx);
      free(shared_words);
      free(cpus);
      return 1;
    }
  }

  struct timespec start;
  struct timespec end;
  pthread_barrier_wait(&start_barrier);
  clock_gettime(CLOCK_MONOTONIC, &start);
  pthread_barrier_wait(&stop_barrier);
  clock_gettime(CLOCK_MONOTONIC, &end);

  uint64_t total_touches = 0;
  int exit_code = 0;
  for (int i = 0; i < threads; ++i) {
    void* ret = NULL;
    pthread_join(thread_handles[i], &ret);
    if (ret != NULL) {
      exit_code = 1;
    }
    total_touches += thread_ctx[i].touches;
  }

  uint64_t final_checksum = 0;
  if (exit_code == 0) {
    for (uint64_t line = 0; line < line_count; ++line) {
      final_checksum += atomic_load_explicit(
          &shared_words[line * (kCacheLineBytes / sizeof(uint64_t))],
          memory_order_relaxed);
    }

    printf("{\"mode\":\"write-shared\",\"threads\":%d,\"cpus\":[", threads);
    for (int i = 0; i < threads; ++i) {
      if (i != 0) {
        printf(",");
      }
      printf("%d", cpus[i]);
    }
    printf("],\"workset_bytes\":%llu,\"iters\":%llu,\"line_count\":%llu,"
           "\"elapsed_sec\":%.9f,\"total_touches\":%llu,"
           "\"final_checksum\":%llu}\n",
           (unsigned long long)workset_bytes,
           (unsigned long long)iters,
           (unsigned long long)line_count,
           elapsed_sec(&start, &end),
           (unsigned long long)total_touches,
           (unsigned long long)final_checksum);
  }

  pthread_barrier_destroy(&start_barrier);
  pthread_barrier_destroy(&stop_barrier);
  free(thread_handles);
  free(thread_ctx);
  free(shared_words);
  free(cpus);
  return exit_code;
}
