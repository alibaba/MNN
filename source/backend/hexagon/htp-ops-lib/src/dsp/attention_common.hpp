static inline size_t attn_align_128(size_t offset) {
  return (offset + 127) & ~((size_t)127);
}

static inline void sync_attention_matmul_lock_acquire() {
  hmx_unit_acquire();
}

static inline void sync_attention_matmul_lock_release() {
  hmx_unit_release();
}

static inline void sync_attention_hmx_section_begin() {
  sync_attention_matmul_lock_acquire();
#if MNN_ATTENTION_HMX_ENABLE_PER_MATMUL
  hmx_manager_enable_execution();
#endif
}

static inline void sync_attention_hmx_section_end() {
#if MNN_ATTENTION_HMX_ENABLE_PER_MATMUL
  hmx_manager_disable_execution();
#endif
  sync_attention_matmul_lock_release();
}

static inline void attn_prepare_dma_desc_2d(dma_desc_2d_t* desc, const void* src, void* dst, uint32_t width, uint32_t height,
                                            uint32_t src_stride, uint32_t dst_stride, uint32_t next) {
  memset(desc, 0, sizeof(dma_desc_2d_t));
  desc->next = next;
  desc->type = DMA_DESC_TYPE_2D;
  desc->src_bypass = 0;
  desc->dst_bypass = 1;
  desc->ordered = 1;
  desc->dstate = DMA_DESC_DSTATE_PENDING;
  desc->src = (uint32_t)src;
  desc->dst = (uint32_t)dst;
  desc->roi_width = width;
  desc->roi_height = height;
  desc->src_stride = src_stride;
  desc->dst_stride = dst_stride;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-optimized"
static inline dma_desc_1d_t* attn_prepare_chained_dma_desc_2d(dma_desc_1d_t* current_descs, dma_desc_2d_t* desc, const void* src,
                                                              void* dst, uint32_t width, uint32_t height, uint32_t src_stride,
                                                              uint32_t dst_stride) {
  attn_prepare_dma_desc_2d(desc, src, dst, width, height, src_stride, dst_stride, 0);
  uintptr_t current_descs_addr = (uintptr_t)current_descs;
  if (current_descs_addr != 0) {
    ((dma_desc_1d_t*)current_descs_addr)->next = (uint32_t)desc;
  }
  return (dma_desc_1d_t*)desc;
}
#pragma clang diagnostic pop

static inline dma_desc_1d_t* attn_prepare_weight_dma_descs(dma_desc_1d_t* current_descs, dma_desc_2d_t* weight_descs,
                                                           const uint8_t* b, __fp16* vtcm_weight_buf, int oy_start, int oy_end,
                                                           int kp, int weightLayoutType, int kv_head, int n_kv_heads, int k_icP,
                                                           int v_ocP) {
  int weight_desc_count = 0;
  if (weightLayoutType == ATTN_HMX_WEIGHT_LAYOUT_K_BLOCK256) {
    int weight_dma_count = oy_end - oy_start;
    const __fp16* weightSrc = (const __fp16*)b + attn_hmx_k_tile_index(oy_start, kv_head, 0, n_kv_heads, k_icP) * 1024;
    current_descs = attn_prepare_chained_dma_desc_2d(current_descs, &weight_descs[weight_desc_count], weightSrc, vtcm_weight_buf,
                                                     16 * kp * 64 * sizeof(int16_t), weight_dma_count,
                                                     k_icP * 1024 * sizeof(__fp16), 1024 * kp * sizeof(int16_t));
    // K pages may have just been written by DSP push_kv; let DMA allocate source lines for coherent QK reads.
    weight_descs[weight_desc_count].cache_alloc = DMA_DESC_CACHEALLOC_READONLY;
    ++weight_desc_count;
  } else {
    int weight_dma_count = oy_end - oy_start;
    for (int k_tile_base = 0; k_tile_base < kp; k_tile_base += ATTN_HMX_KV_BLOCK_TILES) {
      int tiles = kp - k_tile_base;
      if (tiles > ATTN_HMX_KV_BLOCK_TILES) tiles = ATTN_HMX_KV_BLOCK_TILES;
      const __fp16* weightSrc = (const __fp16*)b + attn_hmx_v_tile_index(oy_start, k_tile_base, kv_head, n_kv_heads, v_ocP) * 1024;
      current_descs = attn_prepare_chained_dma_desc_2d(current_descs, &weight_descs[weight_desc_count], weightSrc,
                                                       vtcm_weight_buf + k_tile_base * 1024,
                                                       (uint32_t)(tiles * 1024 * sizeof(int16_t)), weight_dma_count,
                                                       ATTN_HMX_KV_BLOCK_TILES * 1024 * 2, 1024 * kp * 2);
      ++weight_desc_count;
    }
  }
  return current_descs;
}

static void attn_transform_activation_hmx(__fp16* dst, const __fp16* src_raw, int valid_xi, int pair_packs, int kp) {
  int raw_row_stride = pair_packs * 64;
  for (int pair_idx = 0; pair_idx < pair_packs; ++pair_idx) {
    uint8_t* dst_addr_k0 = (uint8_t*)(dst + (pair_idx * 2 + 0) * 32 * 32);
    uint8_t* dst_addr_k1 = (uint8_t*)(dst + (pair_idx * 2 + 1) * 32 * 32);
    int has_k1 = (pair_idx * 2 + 1 < kp);
    int src_col_offset = pair_idx * 64;

    int i = 0;
    for (; i <= valid_xi - 2; i += 2) {
      const uint8_t* src_addr_0 = (const uint8_t*)(src_raw + (size_t)i * raw_row_stride + src_col_offset);
      const uint8_t* src_addr_1 = (const uint8_t*)(src_raw + (size_t)(i + 1) * raw_row_stride + src_col_offset);
      HVX_Vector v0 = vmem(src_addr_0);
      HVX_Vector v1 = vmem(src_addr_1);
      HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, 64);
      vmem(dst_addr_k0 + i * 64) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
      if (has_k1) {
        vmem(dst_addr_k1 + i * 64) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
      }
    }
    if (i < valid_xi) {
      const uint8_t* src_addr_0 = (const uint8_t*)(src_raw + (size_t)i * raw_row_stride + src_col_offset);
      HVX_Vector v0 = vmem(src_addr_0);
      HVX_Vector v1 = Q6_V_vzero();
      HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, 64);
      vmem(dst_addr_k0 + i * 64) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
      if (has_k1) {
        vmem(dst_addr_k1 + i * 64) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
      }
      i += 2;
    }
  }
}

static inline void attn_store_output_packed_fp16(uint8_t* c, __fp16* vtcm_output, int oy_start, int oy_end, int ox, int M,
                                                 int valid_xi, int output_stride, int output_row_offset) {
  const int pack = 64;
  int m_stride = output_stride > 0 ? output_stride : M;
  HVX_VectorPred q_even = Q6_Q_vsetq_R(64);
  int xi_limit = valid_xi & ~1;
  int oy = oy_start;
  for (; oy <= oy_end - 2; oy += 2) {
    __fp16* vtcm_dst0 = vtcm_output + (oy - oy_start) * 1024;
    __fp16* vtcm_dst1 = vtcm_output + (oy + 1 - oy_start) * 1024;
    HVX_Vector* src_ptr0 = (HVX_Vector*)vtcm_dst0;
    HVX_Vector* src_ptr1 = (HVX_Vector*)vtcm_dst1;
    int pack_idx = (oy * 32) / pack;
    uint8_t* dst_ptr = c + (pack_idx * m_stride + output_row_offset + ox * 32) * 128;
    int xi = 0;
    for (; xi < xi_limit; xi += 2) {
      HVX_Vector vLoad0 = Q6_Vh_vdeal_Vh(*src_ptr0++);
      HVX_Vector vLoad1 = Q6_Vh_vdeal_Vh(*src_ptr1++);
      HVX_Vector vLoad0_rot = Q6_V_valign_VVR(vLoad0, vLoad0, 64);
      HVX_Vector vLoad1_rot = Q6_V_valign_VVR(vLoad1, vLoad1, 64);
      vmem(dst_ptr) = Q6_V_vmux_QVV(q_even, vLoad0, vLoad1_rot);
      vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q_even, vLoad0_rot, vLoad1);
      dst_ptr += 256;
    }
    if (xi < valid_xi) {
      HVX_Vector vLoad0 = Q6_Vh_vdeal_Vh(*src_ptr0++);
      HVX_Vector vLoad1 = Q6_Vh_vdeal_Vh(*src_ptr1++);
      HVX_Vector vLoad1_rot = Q6_V_valign_VVR(vLoad1, vLoad1, 64);
      vmem(dst_ptr) = Q6_V_vmux_QVV(q_even, vLoad0, vLoad1_rot);
    }
  }
  if (oy < oy_end) {
    __fp16* vtcm_dst = vtcm_output + (oy - oy_start) * 1024;
    HVX_Vector* src_ptr = (HVX_Vector*)vtcm_dst;
    int pack_idx = (oy * 32) / pack;
    int pack_inner = (oy * 32) % pack;
    uint8_t* dst_ptr = c + (pack_idx * m_stride + output_row_offset + ox * 32) * 128;
    HVX_VectorPred q = pack_inner == 0 ? q_even : Q6_Q_not_Q(q_even);
    int xi = 0;
    for (; xi < xi_limit; xi += 2) {
      HVX_Vector vLoad = Q6_Vh_vdeal_Vh(*src_ptr++);
      HVX_Vector vLoad_rot = Q6_V_valign_VVR(vLoad, vLoad, 64);
      HVX_Vector vFirst = pack_inner == 0 ? vLoad : vLoad_rot;
      HVX_Vector vSecond = pack_inner == 0 ? vLoad_rot : vLoad;
      vmem(dst_ptr) = Q6_V_vmux_QVV(q, vFirst, vmem(dst_ptr));
      vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q, vSecond, vmem(dst_ptr + 128));
      dst_ptr += 256;
    }
    if (xi < valid_xi) {
      HVX_Vector vLoad = Q6_Vh_vdeal_Vh(*src_ptr++);
      if (pack_inner != 0) {
        vLoad = Q6_V_valign_VVR(vLoad, vLoad, 64);
      }
      vmem(dst_ptr) = Q6_V_vmux_QVV(q, vLoad, vmem(dst_ptr));
    }
  }
}

static inline void attn_store_output_linear_fp32(uint8_t* c, __fp16* vtcm_output, int oy_start, int oy_end, int ox, int N,
                                                 int valid_xi, int output_stride, int output_row_offset) {
  int c_stride = output_stride > 0 ? output_stride : N;
  int xi_limit = valid_xi & ~1;
  int oy = oy_start;
  for (; oy <= oy_end - 2; oy += 2) {
    __fp16* vtcm_dst0 = vtcm_output + (oy - oy_start) * 1024;
    __fp16* vtcm_dst1 = vtcm_output + (oy + 1 - oy_start) * 1024;
    HVX_Vector* src_ptr0 = (HVX_Vector*)vtcm_dst0;
    HVX_Vector* src_ptr1 = (HVX_Vector*)vtcm_dst1;
    int col_offset0 = oy * 32;
    int col_offset1 = (oy + 1) * 32;
    int xi = 0;
    for (; xi < xi_limit; xi += 2) {
      HVX_Vector vLoad0 = Q6_Vh_vdeal_Vh(*src_ptr0++);
      HVX_Vector vLoad1 = Q6_Vh_vdeal_Vh(*src_ptr1++);
      float* row0_ptr0 = ((float*)c) + (size_t)(output_row_offset + ox * 32 + xi) * c_stride + col_offset0;
      float* row1_ptr0 = ((float*)c) + (size_t)(output_row_offset + ox * 32 + xi + 1) * c_stride + col_offset0;
      float* row0_ptr1 = ((float*)c) + (size_t)(output_row_offset + ox * 32 + xi) * c_stride + col_offset1;
      float* row1_ptr1 = ((float*)c) + (size_t)(output_row_offset + ox * 32 + xi + 1) * c_stride + col_offset1;

      HVX_VectorPair vLoad0_sf = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(vLoad0));
      HVX_VectorPair vLoad1_sf = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(vLoad1));
      HVX_Vector vRow00 = Q6_V_lo_W(vLoad0_sf);
      HVX_Vector vRow10 = Q6_V_hi_W(vLoad0_sf);
      HVX_Vector vRow01 = Q6_V_lo_W(vLoad1_sf);
      HVX_Vector vRow11 = Q6_V_hi_W(vLoad1_sf);
      vmemu(row0_ptr0) = vRow00;
      vmemu(row1_ptr0) = vRow10;
      vmemu(row0_ptr1) = vRow01;
      vmemu(row1_ptr1) = vRow11;
    }
    if (xi < valid_xi) {
      HVX_Vector vLoad0 = Q6_Vh_vdeal_Vh(*src_ptr0++);
      HVX_Vector vLoad1 = Q6_Vh_vdeal_Vh(*src_ptr1++);
      float* row0_ptr0 = ((float*)c) + (size_t)(output_row_offset + ox * 32 + xi) * c_stride + col_offset0;
      float* row0_ptr1 = ((float*)c) + (size_t)(output_row_offset + ox * 32 + xi) * c_stride + col_offset1;
      HVX_VectorPair vLoad0_sf = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(vLoad0));
      HVX_VectorPair vLoad1_sf = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(vLoad1));
      HVX_Vector vRow00 = Q6_V_lo_W(vLoad0_sf);
      HVX_Vector vRow01 = Q6_V_lo_W(vLoad1_sf);
      vmemu(row0_ptr0) = vRow00;
      vmemu(row0_ptr1) = vRow01;
    }
  }
  if (oy < oy_end) {
    __fp16* vtcm_dst = vtcm_output + (oy - oy_start) * 1024;
    HVX_Vector* src_ptr = (HVX_Vector*)vtcm_dst;
    int col_offset = oy * 32;
    int xi = 0;
    for (; xi < xi_limit; xi += 2) {
      HVX_Vector vLoad = Q6_Vh_vdeal_Vh(*src_ptr++);
      float* row0_ptr = ((float*)c) + (size_t)(output_row_offset + ox * 32 + xi) * c_stride + col_offset;
      float* row1_ptr = ((float*)c) + (size_t)(output_row_offset + ox * 32 + xi + 1) * c_stride + col_offset;
      HVX_VectorPair vLoad_sf = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(vLoad));
      HVX_Vector vRow0 = Q6_V_lo_W(vLoad_sf);
      HVX_Vector vRow1 = Q6_V_hi_W(vLoad_sf);
      vmemu(row0_ptr) = vRow0;
      vmemu(row1_ptr) = vRow1;
    }
    if (xi < valid_xi) {
      HVX_Vector vLoad = Q6_Vh_vdeal_Vh(*src_ptr++);
      float* row0_ptr = ((float*)c) + (size_t)(output_row_offset + ox * 32 + xi) * c_stride + col_offset;
      HVX_VectorPair vLoad_sf = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(vLoad));
      HVX_Vector vRow0 = Q6_V_lo_W(vLoad_sf);
      vmemu(row0_ptr) = vRow0;
    }
  }
}

static inline void run_locked_attn_hmx_matmul(uint8_t* c, const uint8_t* a, const uint8_t* b, int M, int K, int N, int max_K,
                                              int a_stride, int outputLayoutType, float outputScale, int weightLayoutType,
                                              int kv_head, int n_kv_heads) {
  run_locked_attn_hmx_matmul_ex(c, a, b, M, K, N, max_K, a_stride, outputLayoutType, outputScale, weightLayoutType, kv_head,
                                n_kv_heads, 0, 0);
}

static inline void run_locked_attn_hmx_matmul_ex(uint8_t* c, const uint8_t* a, const uint8_t* b, int M, int K, int N,
                                                 int max_K, int a_stride, int outputLayoutType, float outputScale,
                                                 int weightLayoutType, int kv_head, int n_kv_heads, int output_stride,
                                                 int output_row_offset) {
  sync_attention_hmx_section_begin();
  attn_hmx_matmul(c, a, b, M, K, N, max_K, a_stride, outputLayoutType, outputScale, weightLayoutType, kv_head, n_kv_heads,
                  output_stride, output_row_offset);
  sync_attention_hmx_section_end();
}

static inline void run_locked_attn_hmx_matmul_pages_qk(const SyncAttentionTaskState* state, float* scores,
                                                       const __fp16* q_ptr, int rows, int q_stride, int q_row_offset,
                                                       int kv_head, int valid_end) {
  sync_attention_hmx_section_begin();
  attn_hmx_matmul_pages_qk(state, scores, q_ptr, rows, q_stride, q_row_offset, kv_head, valid_end);
  sync_attention_hmx_section_end();
}

static inline void run_locked_attn_hmx_matmul_pages_sv(const SyncAttentionTaskState* state, __fp16* dst, __fp16* temp_O,
                                                       const __fp16* linear_S, int rows, int output_stride, int row_offset,
                                                       int kv_head, int valid_end) {
  sync_attention_hmx_section_begin();
  attn_hmx_matmul_pages_sv(state, dst, temp_O, linear_S, rows, output_stride, row_offset, kv_head, valid_end);
  sync_attention_hmx_section_end();
}

static inline void sync_attention_wait_async_push(const SyncAttentionTaskState* state) {
  AsyncPushKVPagesState* push = state->async_push;
  if (push == NULL) {
    return;
  }
  while (!push->done) {
    asm volatile("pause(#8)" ::: "memory");
  }
}

static inline void sync_attention_add_mask(float* row_scores, const float* mask_ptr, int N, int mask_start_pos) {
  float* row_mask_scores = row_scores + mask_start_pos;
  int valid_mask_len = N - mask_start_pos;
  for (; valid_mask_len >= 32; valid_mask_len -= 32, row_mask_scores += 32, mask_ptr += 32) {
    HVX_Vector v_mask = vmemu(mask_ptr);
    HVX_Vector v_score = vmemu(row_mask_scores);
    v_score = Q6_Vsf_vadd_VsfVsf(v_score, v_mask);
    vmemu(row_mask_scores) = v_score;
  }
  for (; valid_mask_len > 0; --valid_mask_len, ++row_mask_scores, ++mask_ptr) {
    *row_mask_scores += *mask_ptr;
  }
}

static inline float sync_attention_max_f32(const float* row_scores, int N) {
  if (N <= 0) {
    return -INFINITY;
  }
  float neg_inf = -INFINITY;
  HVX_Vector v_max = Q6_V_vsplat_R(*(int*)&neg_inf);
  int i = 0;
  for (; i <= N - 32; i += 32) {
    HVX_Vector v_score = vmemu(row_scores + i);
    v_max = Q6_Vsf_vmax_VsfVsf(v_max, v_score);
  }
  int tail = N - i;
  if (tail > 0) {
    HVX_VectorPred q_tail = Q6_Q_vsetq_R(tail * (int)sizeof(float));
    HVX_Vector v_score = vmemu(row_scores + i);
    v_score = Q6_V_vmux_QVV(q_tail, v_score, v_max);
    v_max = Q6_Vsf_vmax_VsfVsf(v_max, v_score);
  }
  v_max = Q6_Vsf_vmax_VsfVsf(v_max, Q6_V_vror_VR(v_max, 64));
  v_max = Q6_Vsf_vmax_VsfVsf(v_max, Q6_V_vror_VR(v_max, 32));
  v_max = Q6_Vsf_vmax_VsfVsf(v_max, Q6_V_vror_VR(v_max, 16));
  v_max = Q6_Vsf_vmax_VsfVsf(v_max, Q6_V_vror_VR(v_max, 8));
  v_max = Q6_Vsf_vmax_VsfVsf(v_max, Q6_V_vror_VR(v_max, 4));
  float max_arr[32] __attribute__((aligned(128)));
  *(HVX_Vector*)max_arr = v_max;
  return max_arr[0];
}

static inline float sync_attention_exp_and_sum(float* row_scores, int N, float max_value) {
  const float log2e = 1.4426950408889634f;
  HVX_Vector v_log2e = Q6_V_vsplat_R(*(const int*)&log2e);
  HVX_Vector v_max = Q6_V_vsplat_R(*(const int*)&max_value);
  int k_j = 0;
  HVX_Vector v_sum = Q6_V_vzero();
  for (; k_j <= N - 32; k_j += 32) {
    HVX_Vector v_score = vmemu((float*)&row_scores[k_j]);
    v_score = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v_score, v_max));
    v_score = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_score, v_log2e));
    HVX_Vector v_exp = hvx_my_exp2_vsf(v_score);
    vmemu((float*)&row_scores[k_j]) = v_exp;
    v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, v_exp));
  }
  int tail = N - k_j;
  if (tail > 0) {
    HVX_VectorPred q_tail = Q6_Q_vsetq_R(tail * (int)sizeof(float));
    HVX_Vector v_score = vmemu((float*)&row_scores[k_j]);
    v_score = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v_score, v_max));
    v_score = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_score, v_log2e));
    HVX_Vector v_exp = hvx_my_exp2_vsf(v_score);
    v_exp = Q6_V_vmux_QVV(q_tail, v_exp, Q6_V_vzero());
    vmemu((float*)&row_scores[k_j]) = v_exp;
    v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, v_exp));
  }
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 64)));
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 32)));
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 16)));
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 8)));
  v_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_sum, Q6_V_vror_VR(v_sum, 4)));

  float sum_arr[32] __attribute__((aligned(128)));
  *(HVX_Vector*)sum_arr = v_sum;
  return sum_arr[0];
}

static inline void sync_attention_store_fp16_tail(__fp16* dst, HVX_Vector value, int elems) {
  if (elems <= 0) {
    return;
  }
  HVX_VectorPred q_tail = Q6_Q_vsetq_R(elems * (int)sizeof(__fp16));
  if ((((uintptr_t)dst) & 127) == 0) {
    Q6_vmem_QRIV(q_tail, (HVX_Vector*)dst, value);
    return;
  }
  __fp16 tmp[64] __attribute__((aligned(128)));
  vmem(tmp) = value;
  memcpy(dst, tmp, (size_t)elems * sizeof(__fp16));
}

static inline void sync_attention_normalize_to_fp16(__fp16* row_s, float* row_scores, int N, float inv_sum) {
  HVX_Vector v_inv_sum = Q6_V_vsplat_R(*(int*)&inv_sum);
  int k_j = 0;
  for (; k_j <= N - 64; k_j += 64) {
    HVX_Vector v_score0 = vmemu((float*)&row_scores[k_j]);
    HVX_Vector v_score1 = vmemu((float*)&row_scores[k_j + 32]);
    v_score0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_score0, v_inv_sum));
    v_score1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_score1, v_inv_sum));
    HVX_Vector v_out = Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(v_score0, v_score1));
    vmemu(row_s + k_j) = v_out;
  }
  int tail = N - k_j;
  if (tail > 0) {
    HVX_Vector v_score0 = vmemu((float*)&row_scores[k_j]);
    HVX_Vector v_score1 = Q6_V_vzero();
    v_score0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_score0, v_inv_sum));
    if (tail > 32) {
      v_score1 = vmemu((float*)&row_scores[k_j + 32]);
      v_score1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_score1, v_inv_sum));
    }
    HVX_Vector v_out = Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(v_score0, v_score1));
    sync_attention_store_fp16_tail(row_s + k_j, v_out, tail);
  }
}

static inline void sync_attention_zero_fp16(__fp16* row_s, int N) {
  int k_j = 0;
  HVX_Vector v_zero = Q6_V_vzero();
  for (; k_j <= N - 64; k_j += 64) {
    vmemu(row_s + k_j) = v_zero;
  }
  for (; k_j < N; ++k_j) {
    row_s[k_j] = (__fp16)0.0f;
  }
}

static inline void sync_attention_zero_packed_output(__fp16* dst, int rows, int output_stride, int row_offset, int head_dim) {
  int packs = head_dim / 64;
  for (int p = 0; p < packs; ++p) {
    __fp16* base = dst + (size_t)p * output_stride * 64 + row_offset * 64;
    for (int r = 0; r < rows; ++r) {
      memset(base + (size_t)r * 64, 0, 64 * sizeof(__fp16));
    }
  }
}

static inline void sync_attention_accumulate_packed_output(__fp16* dst, const __fp16* src, int rows, int output_stride,
                                                           int row_offset, int head_dim) {
  int packs = head_dim / 64;
  for (int p = 0; p < packs; ++p) {
    __fp16* dst_base = dst + (size_t)p * output_stride * 64 + row_offset * 64;
    const __fp16* src_base = src + (size_t)p * rows * 64;
    for (int r = 0; r < rows; ++r) {
      __fp16* d = dst_base + (size_t)r * 64;
      const __fp16* s = src_base + (size_t)r * 64;
      for (int i = 0; i < 64; i += 64) {
        HVX_Vector vd = vmemu(d + i);
        HVX_Vector vs = vmemu(s + i);
        vmemu(d + i) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(vd, vs));
      }
    }
  }
}

static inline void sync_attention_add_packed_rows(__fp16* dst, const __fp16* src, int rows, int head_dim) {
  int packs = head_dim / 64;
  for (int p = 0; p < packs; ++p) {
    __fp16* d_base = dst + (size_t)p * rows * 64;
    const __fp16* s_base = src + (size_t)p * rows * 64;
    for (int r = 0; r < rows; ++r) {
      __fp16* d = d_base + (size_t)r * 64;
      const __fp16* s = s_base + (size_t)r * 64;
      HVX_Vector vd = vmemu(d);
      HVX_Vector vs = vmemu(s);
      vmemu(d) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(vd, vs));
    }
  }
}

static inline void sync_attention_page_offsets(const SyncAttentionTaskState* state, uint8_t* worker_workspace,
                                               float** scores, __fp16** linear_S, __fp16** temp_O, int rows) {
  size_t offset = 0;
  *scores = (float*)(worker_workspace + offset);
  offset += (size_t)rows * state->N_padded * sizeof(float);
  offset = attn_align_128(offset);
  *linear_S = (__fp16*)(worker_workspace + offset);
  offset += (size_t)rows * state->N_padded * sizeof(__fp16);
  offset = attn_align_128(offset);
  *temp_O = (__fp16*)(worker_workspace + offset);
}

static inline void sync_attention_page_block_offsets(const SyncAttentionTaskState* state, uint8_t* worker_workspace,
                                                     float** scores, __fp16** linear_S, __fp16** accum_O,
                                                     __fp16** temp_O, float** row_max, float** row_sum,
                                                     float** row_scale, int rows) {
  int block_padded = (state->online_block_size + 31) & ~31;
  size_t offset = 0;
  *scores = (float*)(worker_workspace + offset);
  offset += (size_t)rows * block_padded * sizeof(float);
  offset = attn_align_128(offset);
  *linear_S = (__fp16*)(worker_workspace + offset);
  offset += (size_t)rows * block_padded * sizeof(__fp16);
  offset = attn_align_128(offset);
  *accum_O = (__fp16*)(worker_workspace + offset);
  offset += (size_t)rows * state->K_dim_padded * sizeof(__fp16);
  offset = attn_align_128(offset);
  *temp_O = (__fp16*)(worker_workspace + offset);
  offset += (size_t)rows * state->K_dim_padded * sizeof(__fp16);
  offset = attn_align_128(offset);
  *row_max = (float*)(worker_workspace + offset);
  offset += (size_t)rows * sizeof(float);
  *row_sum = (float*)(worker_workspace + offset);
  offset += (size_t)rows * sizeof(float);
  *row_scale = (float*)(worker_workspace + offset);
}

static inline uint8_t* sync_attention_worker_workspace(const SyncAttentionTaskState* state, int worker_index) {
  return state->workspace_base + (size_t)worker_index * state->worker_workspace_bytes;
}

static inline void sync_attention_workspace_offsets(const SyncAttentionTaskState* state, uint8_t* worker_workspace,
                                                    int rows, float** scores, __fp16** linear_S, __fp16** temp_O) {
  if (state->page_count > 0) {
    sync_attention_page_offsets(state, worker_workspace, scores, linear_S, temp_O, rows);
    return;
  }
  size_t offset = 0;
  *scores = (float*)(worker_workspace + offset);
  offset += (size_t)rows * state->N_padded * sizeof(float);
  offset = attn_align_128(offset);
  *linear_S = (__fp16*)(worker_workspace + offset);
  if (temp_O != NULL) {
    *temp_O = NULL;
  }
}

static inline void sync_attention_reset_online_rows(float* row_max, float* row_sum, float* row_scale, int rows) {
  for (int r = 0; r < rows; ++r) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0f;
    row_scale[r] = 0.0f;
  }
}

static inline void sync_attention_run_page_qk(const SyncAttentionTaskState* state, float* scores, const __fp16* q_ptr,
                                              int rows, int q_stride, int q_row_offset, int kv_head, int valid_end) {
  run_locked_attn_hmx_matmul_pages_qk(state, scores, q_ptr, rows, q_stride, q_row_offset, kv_head, valid_end);
}

static inline void sync_attention_run_page_sv(const SyncAttentionTaskState* state, __fp16* dst, __fp16* temp_O,
                                              const __fp16* linear_S, int rows, int output_stride, int row_offset,
                                              int kv_head, int valid_end) {
  run_locked_attn_hmx_matmul_pages_sv(state, dst, temp_O, linear_S, rows, output_stride, row_offset, kv_head, valid_end);
}

static inline void sync_attention_run_page_qk_block(const SyncAttentionTaskState* state, float* scores,
                                                    const __fp16* q_ptr, int rows, int q_stride, int kv_head,
                                                    int page_begin, int page_end, int block_start, int block_valid) {
  sync_attention_hmx_section_begin();
  attn_hmx_matmul_page_qk_block(state, scores, q_ptr, rows, q_stride, kv_head, page_begin, page_end,
                                block_start, block_valid);
  sync_attention_hmx_section_end();
}

static inline void sync_attention_run_page_sv_block(const SyncAttentionTaskState* state, __fp16* dst,
                                                    __fp16* page_temp_O, const __fp16* linear_S, int rows,
                                                    int kv_head, int page_begin, int page_end, int block_start,
                                                    int block_valid) {
  sync_attention_hmx_section_begin();
  attn_hmx_matmul_page_sv_block(state, dst, page_temp_O, linear_S, rows, kv_head, page_begin, page_end,
                                block_start, block_valid);
  sync_attention_hmx_section_end();
}

static inline float sync_attention_load_page_v(const SyncAttentionTaskState* state, int token, int h_kv, int dim) {
  int page = token / state->page_size;
  int page_token = token - page * state->page_size;
  int seq_tile = page_token / 32;
  int seq_inner = page_token % 32;
  int seq_pair = seq_inner / 2;
  int seq_lane = seq_inner & 1;
  int dim_tile = dim / 32;
  int dim_inner = dim & 31;
  const __fp16* v_page = (const __fp16*)state->pastVPages[page];
  const __fp16* v_ptr = v_page + attn_hmx_v_tile_index(dim_tile, seq_tile, h_kv, state->n_kv_heads, state->v_ocP) * 1024 +
                        seq_pair * 64 + seq_lane + dim_inner * 2;
  return (float)(*v_ptr);
}

static inline void sync_attention_store_output_value(const SyncAttentionTaskState* state, int head_id, int q, int dim,
                                                     float value) {
  int pack = dim / 64;
  int inner = dim & 63;
  __fp16* head_O = state->O + (size_t)head_id * (state->head_dim / 64) * state->qo_len * 64;
  head_O[(size_t)pack * state->qo_len * 64 + q * 64 + inner] = (__fp16)value;
}

static inline int sync_attention_try_page_causal_len2(const SyncAttentionTaskState* state, int head_id, int worker_index) {
  if (state->page_count <= 0 || state->mask_stride >= 0 || state->qo_len != 2 || state->seq_current != 0 ||
      state->N != 2 || state->head_dim <= 0 || (state->head_dim % 64) != 0 || state->decode_grouped) {
    return 0;
  }
  const int h_kv = head_id / state->gqa_factor;
  uint8_t* worker_workspace = state->workspace_base + (size_t)worker_index * state->worker_workspace_bytes;
  float* scores = NULL;
  __fp16* linear_S = NULL;
  __fp16* temp_O = NULL;
  sync_attention_page_offsets(state, worker_workspace, &scores, &linear_S, &temp_O, 1);

  for (int d = 0; d < state->head_dim; ++d) {
    sync_attention_store_output_value(state, head_id, 0, d, sync_attention_load_page_v(state, 0, h_kv, d));
  }

  sync_attention_run_page_qk(state, scores, state->Q + state->qo_stride + (size_t)head_id * state->head_dim,
                             1, state->qo_stride, 0, h_kv, 2);
  float max_score = scores[0] > scores[1] ? scores[0] : scores[1];
  float e0 = expf(scores[0] - max_score);
  float e1 = expf(scores[1] - max_score);
  float inv_sum = 1.0f / (e0 + e1);
  float w0 = e0 * inv_sum;
  float w1 = e1 * inv_sum;
  for (int d = 0; d < state->head_dim; ++d) {
    float value = w0 * sync_attention_load_page_v(state, 0, h_kv, d) +
                  w1 * sync_attention_load_page_v(state, 1, h_kv, d);
    sync_attention_store_output_value(state, head_id, 1, d, value);
  }
  (void)linear_S;
  (void)temp_O;
  return 1;
}
