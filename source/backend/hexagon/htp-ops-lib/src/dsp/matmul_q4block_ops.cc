#include "AEEStdDef.h"
#include "HAP_farf.h"

#include "dsp/ops.h"

extern "C" {

AEEResult htp_ops_matmul_q4block_a16_fp16(
                               uint8_t* output, uint8_t* activation,
                               uint8_t* weight, uint8_t* bias,
                               int32 m, int32 k, int32 n,
                               int32 weight_type, int32 layout_type,
                               int32 mp, int32 np, int32 kp,
                               int32 scale_block_num,
                               int32 scale_asymmetric) {
  (void) layout_type;  // currently only layout_type == 1 (permuted) is supported
  (void) weight_type;
  (void) scale_asymmetric;

  int icP = (k + 31) / 32;
  int ocP = (n + 31) / 32;
  const uint8_t *b_scale = weight + icP * ocP * 32 * 16;

  int mm_ret = 0;
  if (m == 1) {
    mm_ret = hmx_matmulq4blockfp16_mle32(output, activation, weight, b_scale, bias,
                                         m, k, n, mp, np, kp, scale_block_num, 0);
  } else if (m <= 32) {
    mm_ret = hmx_matmulq4fp16_mle32(output, activation, weight, b_scale, bias,
                                    m, k, n, mp, np, kp, scale_block_num, 0);
  } else {
    mm_ret = hmx_matmulq4fp16(output, activation, weight, b_scale, bias,
                              m, k, n, mp, np, kp, scale_block_num, 0);
  }
  if (mm_ret != 0) {
    FARF(ALWAYS, "block q4 matmul failed: %d", mm_ret);
    return mm_ret;
  }
  return 0;
}

}  // extern "C"
