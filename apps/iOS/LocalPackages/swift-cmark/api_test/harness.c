// _GNU_SOURCE is all ISO/POSIX/XOPEN/BSD/SVID + GNU extensions. It also sets
// _DEFAULT_SOURCE on newer glibc. We need this for strdup/snprintf/fdopen/etc.
#define _GNU_SOURCE

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "harness.h"

test_batch_runner *test_batch_runner_new() {
  return (test_batch_runner *)calloc(1, sizeof(test_batch_runner));
}

static void test_result(test_batch_runner *runner, int cond, const char *msg,
                        va_list ap) {
  ++runner->test_num;

  if (cond) {
    ++runner->num_passed;
  } else {
    fprintf(stderr, "FAILED test %d: ", runner->test_num);
    vfprintf(stderr, msg, ap);
    fprintf(stderr, "\n");
    ++runner->num_failed;
  }
}

void SKIP(test_batch_runner *runner, int num_tests) {
  runner->test_num += num_tests;
  runner->num_skipped += num_tests;
}

void OK(test_batch_runner *runner, int cond, const char *msg, ...) {
  va_list ap;
  va_start(ap, msg);
  test_result(runner, cond, msg, ap);
  va_end(ap);
}

void INT_EQ(test_batch_runner *runner, int got, int expected, const char *msg,
            ...) {
  int cond = got == expected;

  va_list ap;
  va_start(ap, msg);
  test_result(runner, cond, msg, ap);
  va_end(ap);

  if (!cond) {
    fprintf(stderr, "  Got:      %d\n", got);
    fprintf(stderr, "  Expected: %d\n", expected);
  }
}

#if !defined(_WIN32) && !defined(__wasi__)
#include <unistd.h>

static char *write_tmp(char const *header, char const *data) {
  char *name = strdup("/tmp/fileXXXXXX");
  int fd = mkstemp(name);
  FILE *f = fdopen(fd, "w+");
  fputs(header, f);
  fwrite(data, 1, strlen(data), f);
  fclose(f);
  return name;
}

#endif

void STR_EQ(test_batch_runner *runner, const char *got, const char *expected,
            const char *msg, ...) {
  int cond = strcmp(got, expected) == 0;

  va_list ap;
  va_start(ap, msg);
  test_result(runner, cond, msg, ap);
  va_end(ap);

  if (!cond) {
#if !defined(_WIN32) && !defined(__wasi__)
    char *got_fn = write_tmp("actual\n", got);
    char *expected_fn = write_tmp("expected\n", expected);
    char buf[1024];
    snprintf(buf, sizeof(buf), "git diff --no-index %s %s", expected_fn, got_fn);
    system(buf);
    remove(got_fn);
    remove(expected_fn);
    free(got_fn);
    free(expected_fn);
#else
    fprintf(stderr, "  Got:      \"%s\"\n", got);
    fprintf(stderr, "  Expected: \"%s\"\n", expected);
#endif
  }
}

int test_ok(test_batch_runner *runner) { return runner->num_failed == 0; }

void test_print_summary(test_batch_runner *runner) {
  int num_passed = runner->num_passed;
  int num_skipped = runner->num_skipped;
  int num_failed = runner->num_failed;

  fprintf(stderr, "%d tests passed, %d failed, %d skipped\n", num_passed,
          num_failed, num_skipped);

  if (test_ok(runner)) {
    fprintf(stderr, "PASS\n");
  } else {
    fprintf(stderr, "FAIL\n");
  }
}
