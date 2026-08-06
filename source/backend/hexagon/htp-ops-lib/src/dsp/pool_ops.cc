#include "dsp/mmap_mgr.h"
#include <HAP_mem.h>
#include "AEEStdDef.h"
#include "remote.h"
#include "hexagon_protos.h"
#include "hexagon_types.h"
#include "htp_ops.h"
#include "AEEStdErr.h"
#include "qurt_memory.h"
#include "HAP_farf.h"
#include "HAP_perf.h"

#include "dsp/ops.h"

extern "C" {


AEEResult htp_ops_pool2d_fp16(uint8_t* output, uint8_t* input,
                              int32_t batch, int32_t ih, int32_t iw, int32_t oh, int32_t ow,
                              int32_t c4, int32_t kernelY, int32_t kernelX, int32_t strideY, int32_t strideX,
                              int32_t padY, int32_t padX, int32_t padType, int32_t countType, int32_t poolType) {
  __fp16 *outPtr = (__fp16 *) output;
  const __fp16 *inPtr = (const __fp16 *) input;

  size_t inSize = (size_t) batch * ih * iw * c4 * 4 * sizeof(__fp16); (void)inSize;
  size_t outSize = (size_t) batch * oh * ow * c4 * 4 * sizeof(__fp16); (void)outSize;

  int err = hvx_pool2d_fp16(outPtr, inPtr, batch, ih, iw, oh, ow, c4, kernelY, kernelX, strideY, strideX, padY, padX,
                        padType, countType, poolType);

  return err;
}

}  // extern "C"
