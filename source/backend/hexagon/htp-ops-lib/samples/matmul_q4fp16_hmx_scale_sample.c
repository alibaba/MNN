// Reference sample only. This file is intentionally not part of the build.
//
// It shows the working matmul_q4fp16 variant that applies q4 scale and optional
// bias through HMX output scale/bias instead of the active HVX post-processing
// path. Copy the helpers into src/dsp/ops/matmul_q4fp16.c and replace only the
// inner HMX/store loop shown at the bottom if this experiment needs to be
// re-tested.
//
// Observed result on ntp1_onelayer:
//   - Correctness passed: cos_sim about 0.99998 in the previous experiment.
//   - Performance regressed: MATMUL_Q4A16_FP16 about 294.6 ms vs about
//     230.4 ms for the restored HVX post-processing path.

static inline HVX_Vector* q4_hmx_scale_table_for_oy(HVX_Vector* hmx_scales,
                                                    int oy_start,
                                                    int oy) {
  return hmx_scales + (oy - oy_start) * 2;
}

static inline void prepare_q4_hmx_scale_bias_table(HVX_Vector* hmx_scales,
                                                   const uint8_t* vtcm_scales,
                                                   const uint8_t* bias,
                                                   int oy_start,
                                                   int oy) {
  const __fp16* scale_ptr = (const __fp16*)(vtcm_scales + (oy - oy_start) * 64);
  const __fp16* bias_ptr = bias ? (const __fp16*)bias + oy * 32 : NULL;
  hmx_init_per_column_scale_bias(q4_hmx_scale_table_for_oy(hmx_scales, oy_start, oy),
                                 scale_ptr, bias_ptr);
}

static inline void prepare_q4_hmx_scale_bias_range(HVX_Vector* hmx_scales,
                                                   const uint8_t* vtcm_scales,
                                                   const uint8_t* bias,
                                                   int oy_start,
                                                   int oy_end) {
  for (int oy = oy_start; oy < oy_end; ++oy) {
    prepare_q4_hmx_scale_bias_table(hmx_scales, vtcm_scales, bias, oy_start, oy);
  }
}

static inline HVX_Vector post_hmx_scaled_output_vec(HVX_Vector v) {
  return Q6_Vh_vdeal_Vh(v);
}

static inline void store_q4_hmx_scaled_output_tile(uint8_t* c,
                                                   __fp16* vtcm_output,
                                                   int M,
                                                   int ox,
                                                   int oy) {
  const int pack_idx = (oy * 32) / 64;
  const int pack_inner = (oy * 32) & 63;
  int valid_xi = M - ox * 32;
  if (valid_xi > 32) valid_xi = 32;
  if (valid_xi < 0) valid_xi = 0;

  const int xi_limit = valid_xi & ~1;
  HVX_VectorPred q = pack_inner == 0 ? Q6_Q_vsetq_R(64) : Q6_Q_not_Q(Q6_Q_vsetq_R(64));
  HVX_Vector* src_ptr = (HVX_Vector*)vtcm_output;
  uint8_t* dst_ptr = c + (size_t)(pack_idx * M + ox * 32) * 128;

  int xi = 0;
  for (; xi < xi_limit; xi += 2) {
    HVX_Vector v = post_hmx_scaled_output_vec(*src_ptr++);
    HVX_Vector v_rot = Q6_V_valign_VVR(v, v, 64);
    HVX_Vector v_first = pack_inner == 0 ? v : v_rot;
    HVX_Vector v_second = pack_inner == 0 ? v_rot : v;
    vmem(dst_ptr) = Q6_V_vmux_QVV(q, v_first, vmem(dst_ptr));
    vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q, v_second, vmem(dst_ptr + 128));
    dst_ptr += 256;
  }
  if (xi < valid_xi) {
    HVX_Vector v = post_hmx_scaled_output_vec(*src_ptr++);
    if (pack_inner != 0) {
      v = Q6_V_valign_VVR(v, v, 64);
    }
    vmem(dst_ptr) = Q6_V_vmux_QVV(q, v, vmem(dst_ptr));
  }
}

static inline int pack_q4_hmx_scaled_output_tile_pair(uint8_t* packed,
                                                      __fp16* vtcm_output0,
                                                      __fp16* vtcm_output1,
                                                      int M,
                                                      int ox) {
  int valid_xi = M - ox * 32;
  if (valid_xi > 32) valid_xi = 32;
  if (valid_xi < 0) valid_xi = 0;
  const int xi_limit = valid_xi & ~1;
  HVX_VectorPred q_low = Q6_Q_vsetq_R(64);

  HVX_Vector* src0_ptr = (HVX_Vector*)vtcm_output0;
  HVX_Vector* src1_ptr = (HVX_Vector*)vtcm_output1;
  uint8_t* dst_ptr = packed;

  int xi = 0;
  for (; xi < xi_limit; xi += 2) {
    HVX_Vector v0 = post_hmx_scaled_output_vec(*src0_ptr++);
    HVX_Vector v1 = post_hmx_scaled_output_vec(*src1_ptr++);
    HVX_Vector v0_rot = Q6_V_valign_VVR(v0, v0, 64);
    HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
    vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
    vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q_low, v0_rot, v1);
    dst_ptr += 256;
  }
  if (xi < valid_xi) {
    HVX_Vector v0 = post_hmx_scaled_output_vec(*src0_ptr++);
    HVX_Vector v1 = post_hmx_scaled_output_vec(*src1_ptr++);
    HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
    vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
  }
  return valid_xi * 128;
}

// Replace the active matmul_q4fp16 inner loop:
//
//   hmx_unit_acquire();
//   hmx_set_output_scales(vtcm_hmx_scales);
//   for (int oy = oy_start; oy < oy_end; ++oy) {
//     ...
//     hmx_consume_accumulator_fp16(vtcm_output);
//     pack_q4_output_tile_pair(...) or store_q4_output_tile(...);
//   }
//   hmx_unit_release();
//
// with this HMX-scale version. The rest of the function, including DMA and q4
// weight expansion, can stay unchanged.
//
// Before entering the ox loop for an oy chunk, build the HMX scale tables once:
//
//   prepare_q4_hmx_scale_bias_range(vtcm_hmx_scales, vtcm_scales, bias,
//                                   oy_start, oy_end);
//
static inline void matmul_q4fp16_hmx_scale_inner_loop_sample(
    uint8_t* c,
    const __fp16* vtcm_activation,
    __fp16* vtcm_weight,
    __fp16* vtcm_output,
    HVX_Vector* vtcm_hmx_scales,
    const uint8_t* vtcm_scales,
    const uint8_t* bias,
    int M,
    int kp,
    int pack,
    int ox,
    int ox_offset,
    int oy_start,
    int oy_end) {
  hmx_unit_acquire();
  for (int oy = oy_start; oy < oy_end; ++oy) {
#ifndef SIMULATOR_MOCK_HMX
    asm volatile("mxclracc.hf");
#endif
    hmx_set_output_scales(q4_hmx_scale_table_for_oy(vtcm_hmx_scales, oy_start, oy));

    int oy_offset = (oy - oy_start) * 16 * kp;
    hmx_load_q4_tiles(vtcm_activation + ox_offset, vtcm_weight + oy_offset * 64, kp);
    hmx_consume_accumulator_fp16(vtcm_output);

    if (pack == 64 && oy + 1 < oy_end && ((oy * 32) & 63) == 0) {
      __fp16* vtcm_output_next = vtcm_output + 1024;
#ifndef SIMULATOR_MOCK_HMX
      asm volatile("mxclracc.hf");
#endif
      hmx_set_output_scales(q4_hmx_scale_table_for_oy(vtcm_hmx_scales, oy_start, oy + 1));

      int oy_offset_next = (oy + 1 - oy_start) * 16 * kp;
      hmx_load_q4_tiles(vtcm_activation + ox_offset, vtcm_weight + oy_offset_next * 64, kp);
      hmx_consume_accumulator_fp16(vtcm_output_next);

      uint8_t* dst_ptr = c + (size_t)(((oy * 32) / 64) * M + ox * 32) * 128;
      pack_q4_hmx_scaled_output_tile_pair(dst_ptr, vtcm_output, vtcm_output_next, M, ox);
      ++oy;
    } else {
      store_q4_hmx_scaled_output_tile(c, vtcm_output, M, ox, oy);
    }
  }
  hmx_unit_release();
}
