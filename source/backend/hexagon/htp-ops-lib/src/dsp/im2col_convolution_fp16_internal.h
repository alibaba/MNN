#pragma once

#include <stdint.h>
#include "dsp/ops.h"

void hmx_im2col_compute_tile_fp16(const __fp16* act_tile, const __fp16* weight_tile,
                                  int kp, __fp16* vtcm_output);
void hmx_im2col_fill_activation_tiles(__fp16* vtcm_activation, const uint8_t* src,
                                      const Im2ColParameter* p,
                                      int tile_start, int tile_count, int kp, int batch);
