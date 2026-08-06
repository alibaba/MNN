#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <stddef.h>
#include <stdint.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>

#include "dsp/hvx_utils.h"

extern "C" {

static inline uint16_t htp_ops_topk_fp16_max_bits(const __fp16* src, int32_t size) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = size & -vec_len;
  const __fp16* ptr = src;
  int i = 0;

  __fp16 best_scalar = src[0];
  HVX_Vector best_v = Q6_Vh_vsplat_R(((const uint16_t*)src)[0]);
  for (; i < vec_end; i += vec_len) {
    HVX_Vector v = vmemu((const HVX_Vector*)ptr);
    best_v = Q6_Vhf_vmax_VhfVhf(best_v, v);
    ptr += vec_len;
  }

  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 64));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 32));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 16));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 8));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 4));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 2));

  __attribute__((aligned(128))) uint16_t tmp[vec_len];
  vmemu((HVX_Vector*)tmp) = best_v;
  uint16_t best_bits = tmp[0];
  best_scalar = *(__fp16*)&best_bits;

  for (; i < size; ++i) {
    const __fp16 value = src[i];
    if (value > best_scalar) {
      best_scalar = value;
      best_bits = ((const uint16_t*)src)[i];
    }
  }
  return best_bits;
}

AEEResult htp_ops_topkv2_k1_fp16(uint8_t* values, uint8_t* indices, uint8_t* input, int32_t rowSize, int32_t rows) {
  if (values == nullptr || indices == nullptr || input == nullptr || rowSize <= 0 || rows <= 0) {
    return -1;
  }
  const __fp16* src = (const __fp16*)input;
  __fp16* valueOut = (__fp16*)values;
  int32_t* indexOut = (int32_t*)indices;
  for (int r = 0; r < rows; ++r) {
    const __fp16* row = src + (size_t)r * rowSize;
    const uint16_t bestBits = htp_ops_topk_fp16_max_bits(row, rowSize);
    const uint16_t* rowBits = (const uint16_t*)row;
    int32_t bestIndex = 0;
    for (int i = 0; i < rowSize; ++i) {
      if (rowBits[i] == bestBits) {
        bestIndex = i;
        break;
      }
    }
    ((uint16_t*)valueOut)[r] = bestBits;
    indexOut[r] = bestIndex;
  }
  return 0;
}

} // extern "C"
