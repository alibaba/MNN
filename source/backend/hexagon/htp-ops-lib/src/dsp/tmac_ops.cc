#include "AEEStdDef.h"
#include "HAP_farf.h"

#include "dsp/ops.h"

extern "C" {

AEEResult htp_ops_tmac_a16w1_fp16(uint8_t* output, uint8_t* activation,
                                  uint8_t* weight, uint8_t* scale, uint8_t* bias,
                                  int32 m, int32 ic, int32 oc, int32 scale_block_num,
                                  int32 scale_asymmetric, int32 relu, int32 relu6, int32 output_bytes) {
  int ret = scalar_tmac_a16w1_fp16(output, activation, weight, (const float*)scale, bias,
                                   m, ic, oc, scale_block_num, scale_asymmetric, relu, relu6, output_bytes);
  if (ret != 0) {
    FARF(ALWAYS, "tmac a16w1 failed: %d", ret);
    return ret;
  }
  return 0;
}

}  // extern "C"
