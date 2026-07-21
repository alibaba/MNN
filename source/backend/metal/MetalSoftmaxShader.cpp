// Copyright @ MNN
#include "MetalSoftmaxShader.hpp"

namespace MNN {

// Plane Softmax (scalar)
const char* gSoftmaxPlaneSrc = R"metal(
#include <metal_stdlib>
using namespace metal;
struct softmax_shape {
  int inside_size;
  int axis_length;
  int outside_size;
  int flat_length;
};
kernel void softmax_plane(const device T* in [[buffer(0)]],
                          device T* out [[buffer(1)]],
                          constant softmax_shape& s [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
  if ((int)gid.x >= s.inside_size || (int)gid.y >= s.outside_size) return;
  // Long offset: at 24K seq * 151K vocab (LLM LM-head softmax) the product
  // overflows int32.
  const long axis_off = (long)gid.y * s.axis_length * s.inside_size + int(gid.x);
  const device T* axis_in = in + axis_off;
  device T* axis_out = out + axis_off;
  float maxv = -FLT_MAX;
  for (int i = 0; i < s.axis_length; ++i) {
    maxv = max(maxv, float(axis_in[i * s.inside_size]));
  }
  float sumv = 0.0f;
  for (int i = 0; i < s.axis_length; ++i) {
    sumv += exp(float(axis_in[i * s.inside_size]) - maxv);
  }
  for (int i = 0; i < s.axis_length; ++i) {
    axis_out[i * s.inside_size] = (T)(exp(float(axis_in[i * s.inside_size]) - maxv) / sumv);
  }
}
)metal";

// Plane Softmax with simd group reduce (scalar)
const char* gSoftmaxPlaneSgSrc = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct softmax_shape {
  int inside_size;
  int axis_length;
  int outside_size;
  int flat_length;
};
#define SIMD_GROUP_WIDTH 32
kernel void softmax_plane_sg(const device T* in [[buffer(0)]],
                             device T* out [[buffer(1)]],
                             constant softmax_shape& s [[buffer(2)]],
                             uint2 gid [[threadgroup_position_in_grid]],
                             uint  tiisg [[thread_index_in_simdgroup]]) {
  if ((int)gid.x >= s.inside_size || (int)gid.y >= s.outside_size) return;
  const long axis_off = (long)gid.y * s.axis_length * s.inside_size + int(gid.x);
  const device T* axis_in = in + axis_off;
  device T* axis_out = out + axis_off;
  float lmax = -FLT_MAX;
  for (int i = tiisg; i < s.axis_length; i += SIMD_GROUP_WIDTH) {
    lmax = max(lmax, float(axis_in[i * s.inside_size]));
  }
  float maxv = simd_max(lmax);
  float lsum = 0.0f;
  for (int i = tiisg; i < s.axis_length; i += SIMD_GROUP_WIDTH) {
    lsum += exp(float(axis_in[i * s.inside_size]) - maxv);
  }
  float sumv = simd_sum(lsum);
  for (int i = tiisg; i < s.axis_length; i += SIMD_GROUP_WIDTH) {
    axis_out[i * s.inside_size] = (T)(exp(float(axis_in[i * s.inside_size]) - maxv) / sumv);
  }
}
)metal";

// Plane Softmax with multi-simdgroup threadgroup reduction
const char* gSoftmaxPlaneSgTG = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct softmax_shape {
  int inside_size;
  int axis_length;
  int outside_size;
  int flat_length;
};
#define SIMD_GROUP_WIDTH 32
#ifndef TG_SIZE
#define TG_SIZE 128
#endif
#define SG_PER_TG (TG_SIZE / SIMD_GROUP_WIDTH)

kernel void softmax_plane_sg_tg(const device T* in [[buffer(0)]],
                                device T* out [[buffer(1)]],
                                constant softmax_shape& s [[buffer(2)]],
                                uint2 gtp [[threadgroup_position_in_grid]],
                                uint  tiisg [[thread_index_in_simdgroup]],
                                uint  sgitg [[simdgroup_index_in_threadgroup]]) {
  if ((int)gtp.x >= s.inside_size || (int)gtp.y >= s.outside_size) return;
  const long axis_off = (long)gtp.y * s.axis_length * s.inside_size + int(gtp.x);
  const device T* axis_in = in + axis_off;
  device T* axis_out = out + axis_off;

  const int stride = SIMD_GROUP_WIDTH * SG_PER_TG;
  int start = int(tiisg) + int(sgitg) * SIMD_GROUP_WIDTH;

  // 1) Max reduction
  float lmax = -FLT_MAX;
  for (int i = start; i < s.axis_length; i += stride) {
    lmax = max(lmax, float(axis_in[i * s.inside_size]));
  }
  float sgMax = simd_max(lmax);
  threadgroup float tgMax[SG_PER_TG];
  if (tiisg == 0) tgMax[sgitg] = sgMax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float finalMaxStore[1];
  if (sgitg == 0 && tiisg == 0) {
    float fm = -FLT_MAX;
    for (int k = 0; k < SG_PER_TG; ++k) fm = max(fm, tgMax[k]);
    finalMaxStore[0] = fm;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxv = finalMaxStore[0];

  // 2) Sum reduction
  float lsum = 0.0f;
  for (int i = start; i < s.axis_length; i += stride) {
    lsum += exp(float(axis_in[i * s.inside_size]) - maxv);
  }
  float sgSum = simd_sum(lsum);
  threadgroup float tgSum[SG_PER_TG];
  if (tiisg == 0) tgSum[sgitg] = sgSum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float finalSumStore[1];
  if (sgitg == 0 && tiisg == 0) {
    float fs = 0.0f;
    for (int k = 0; k < SG_PER_TG; ++k) fs += tgSum[k];
    finalSumStore[0] = fs;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sumv = finalSumStore[0];

  // 3) Write back
  for (int i = start; i < s.axis_length; i += stride) {
    axis_out[i * s.inside_size] = (T)(exp(float(axis_in[i * s.inside_size]) - maxv) / sumv);
  }
}
)metal";

// Attention variant (uses ftype and axis_align_length)
const char* gSoftmaxSgReduce = R"metal(
#include <metal_stdlib>
using namespace metal;
struct softmax_shape {
    int inside_size;
    int axis_length;
    int outside_size;
    int axis_align_length;
};
#define SIMD_GROUP_WIDTH 32

kernel void softmax_plane(const device ftype *in [[buffer(0)]],
                          device ftype *out [[buffer(1)]],
                          constant softmax_shape& s [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.inside_size || (int)gid.y >= s.outside_size) return;
    // Use long for the outer offset: for LLM attention softmax at 24K+ seq,
    // gid.y * axis_length * inside_size = B*H*seq * seq easily exceeds INT_MAX.
    long in_offset = (long)gid.y * s.axis_length * s.inside_size + gid.x;
    long out_offset = (long)gid.y * s.axis_align_length * s.inside_size + gid.x;
    auto axis_in  = in + in_offset;
    auto axis_out = out + out_offset;
    float max1 = -FLT_MAX;
    for (int i = 0; i < s.axis_length; i++) {
        max1 = max(max1, float(axis_in[i * s.inside_size]));
    }
    float sum1 = 0;
    for (int i = 0; i < s.axis_length; i++) {
        sum1 += exp(float(axis_in[i * s.inside_size]) - float(max1));
    }
    for (int i = 0; i < s.axis_align_length; i++) {
        axis_out[i * s.inside_size] = i >= s.axis_length ? ftype(0.0) : ftype(exp(float(axis_in[i * s.inside_size]) - float(max1)) / sum1);
    }
}

kernel void softmax_plane_sg(const device ftype *in     [[buffer(0)]],
                        device ftype *out          [[buffer(1)]],
                        constant softmax_shape& s   [[buffer(2)]],
                        uint2 gid[[threadgroup_position_in_grid]],
                        uint  tiisg[[thread_index_in_simdgroup]],
                        uint  sgitg[[simdgroup_index_in_threadgroup]]
    ) {
    if ((int)gid.x >= s.inside_size || (int)gid.y >= s.outside_size) return;
    long in_offset = (long)gid.y * s.axis_length * s.inside_size + gid.x;
    long out_offset = (long)gid.y * s.axis_align_length * s.inside_size + gid.x;
    auto axis_in  = in + in_offset;
    auto axis_out = out + out_offset;
    float max1 = -FLT_MAX;
    for (int i = tiisg; i < s.axis_length; i+=SIMD_GROUP_WIDTH) {
        max1 = max(max1, float(axis_in[i * s.inside_size]));
    }
    max1 = simd_max(max1);
    float sum1 = 0;
    for (int i = tiisg; i < s.axis_length; i+=SIMD_GROUP_WIDTH) {
        sum1 += exp(float(axis_in[i * s.inside_size]) - float(max1));
    }
    sum1 = simd_sum(sum1);
    for (int i = tiisg; i < s.axis_align_length; i+=SIMD_GROUP_WIDTH) {
        axis_out[i * s.inside_size] = i >= s.axis_length ? ftype(0.0) : ftype(exp(float(axis_in[i * s.inside_size]) - float(max1)) / sum1);
    }
}

)metal";

}
