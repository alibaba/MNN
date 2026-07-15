#pragma once

#include <stddef.h>

static inline size_t ceil_div(size_t num, size_t den) {
  return (num + den - 1) / den;
}

static inline size_t align_up(size_t v, size_t align) {
  return ceil_div(v, align) * align;
}

static inline size_t align_down(size_t v, size_t align) {
  return (v / align) * align;
}

static inline size_t smax(size_t a, size_t b) {
  return a > b ? a : b;
}

static inline size_t smin(size_t a, size_t b) {
  return a < b ? a : b;
}
