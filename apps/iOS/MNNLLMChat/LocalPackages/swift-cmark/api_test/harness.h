#ifndef CMARK_API_TEST_HARNESS_H
#define CMARK_API_TEST_HARNESS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int test_num;
  int num_passed;
  int num_failed;
  int num_skipped;
} test_batch_runner;

test_batch_runner *test_batch_runner_new();

void SKIP(test_batch_runner *runner, int num_tests);

void OK(test_batch_runner *runner, int cond, const char *msg, ...);

void INT_EQ(test_batch_runner *runner, int got, int expected, const char *msg,
            ...);

void STR_EQ(test_batch_runner *runner, const char *got, const char *expected,
            const char *msg, ...);

int test_ok(test_batch_runner *runner);

void test_print_summary(test_batch_runner *runner);

#ifdef __cplusplus
}
#endif

#endif
