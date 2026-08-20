#include "AEEStdDef.h"
#include "HAP_farf.h"

#include "dsp/ops.h"

extern "C" {

AEEResult htp_ops_matmul_q4block_a16_fp16(uint8_t *output, uint8_t *activation, uint8_t *weight, uint8_t *bias, int32 m,
                                          int32 k, int32 n, int32 weight_type, int32 layout_type, int32 mp, int32 np,
                                          int32 kp, int32 scale_block_num, int32 scale_asymmetric,
                                          int32 weight_is_vrmpy) {
  (void) layout_type;  // currently only layout_type == 1 (permuted) is supported
  (void) weight_type;

  int icP = (k + 31) / 32;
  int ocP = (n + 31) / 32;
  const uint8_t *b_scale = weight + icP * ocP * 32 * 16;

  int mm_ret = 0;
  if (m == 1) {
    mm_ret = hmx_matmulq4blockfp16_mle32(output, activation, weight, b_scale, bias, m, k, n, mp, np, kp,
                                         scale_block_num, scale_asymmetric, weight_is_vrmpy);
  } else if (m <= 32) {
    mm_ret = hmx_matmulq4fp16_mle32(output, activation, weight, b_scale, bias, m, k, n, mp, np, kp, scale_block_num,
                                    scale_asymmetric, weight_is_vrmpy);
  } else {
    mm_ret = hmx_matmulq4fp16(output, activation, weight, b_scale, bias, m, k, n, mp, np, kp, scale_block_num,
                              scale_asymmetric, weight_is_vrmpy);
  }
  if (mm_ret != 0) {
    FARF(ALWAYS, "block q4 matmul failed: %d", mm_ret);
    return mm_ret;
  }
  return 0;
}

// Decode GEMV (M=1) integer vrmpy path. weight buffer layout:
//   [int4 vrmpy weight: icP*ocP*512 bytes][sw fp32: ocP*nblk*32*(asym ? 2 : 1)*4 bytes]
// Each asymmetric scale entry is 32 fp32 scale followed by 32 fp32 qbias.
AEEResult htp_ops_matmul_q4a16_gemv_i8(uint8_t *output, uint8_t *activation, uint8_t *weight, uint8_t *bias, int32 k,
                                       int32 n, int32 scale_block_num, int32 scale_asymmetric) {
  int            icP     = (k + 31) / 32;
  int            ocP     = (n + 31) / 32;
  const uint8_t *b_scale = weight + (size_t) icP * ocP * 512;
  int            mm_ret =
    hmx_matmulq4block_gemv_i8(output, activation, weight, b_scale, bias, k, n, scale_block_num, scale_asymmetric);
  if (mm_ret != 0) {
    FARF(ALWAYS, "q4 gemv i8 matmul failed: %d", mm_ret);
    return mm_ret;
  }
  return 0;
}

}  // extern "C"
