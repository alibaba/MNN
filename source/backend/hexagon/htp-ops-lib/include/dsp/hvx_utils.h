#ifndef MNN_DSP_HVX_UTILS_H
#define MNN_DSP_HVX_UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <hexagon_protos.h>
#include <hexagon_types.h>

#define HVX_INLINE_ALWAYS inline __attribute__((unused, always_inline))

#define vmem(A)  (*((HVX_Vector *)(A)))
#define vmemu(A) (*((HVX_UVector *)(A)))

#ifndef LOG2VLEN
#define LOG2VLEN 7
#endif

#define VLEN       (1 << LOG2VLEN)
#define VLEN_SHORT ((1 << LOG2VLEN) >> 1)
#define VLEN_WORD  ((1 << LOG2VLEN) >> 2)

typedef union {
  HVX_VectorPair VV;
  struct {
    HVX_Vector lo;
    HVX_Vector hi;
  } V;
} HVX_DV;

static HVX_INLINE_ALWAYS int32_t is_aligned(const void *addr, uint32_t align) {
  return (((uintptr_t)addr) & ((uintptr_t)align - 1u)) == 0;
}

static HVX_INLINE_ALWAYS void l2fetch(const void *addr, uint32_t stride, uint32_t width, uint32_t height,
                                      uint32_t dir) {
  uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
  __asm__ __volatile__("l2fetch(%0, %1)" : : "r"(addr), "r"(control));
}

static HVX_INLINE_ALWAYS void vstu_variable(void *addr, uint32_t bytes, HVX_Vector value) {
  if (bytes == 0) {
    return;
  }
  if (bytes > VLEN) {
    bytes = VLEN;
  }
  _Alignas(128) uint8_t temp[VLEN];
  vmem(temp) = value;
  memcpy(addr, temp, bytes);
}

static HVX_INLINE_ALWAYS void vstdu_variable(void *addr, uint32_t bytes, HVX_VectorPair value) {
  if (bytes <= VLEN) {
    vstu_variable(addr, bytes, Q6_V_lo_W(value));
    return;
  }
  vstu_variable(addr, VLEN, Q6_V_lo_W(value));
  vstu_variable((uint8_t *)addr + VLEN, bytes - VLEN, Q6_V_hi_W(value));
}

static HVX_INLINE_ALWAYS uint16_t fp16_to_bits(const __fp16 *value) {
  uint16_t bits = 0;
  memcpy(&bits, value, sizeof(bits));
  return bits;
}

static HVX_INLINE_ALWAYS HVX_Vector vqf16_from_int(HVX_Vector value) {
  HVX_Vector half = Q6_Vhf_equals_Vh(value);
  return Q6_Vqf16_vadd_VhfVhf(half, Q6_V_vzero());
}

#endif
