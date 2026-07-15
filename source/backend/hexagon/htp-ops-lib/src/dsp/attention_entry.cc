#include "attention_private.hpp"

static void preprocess_mask_to_fp32(float* restrict dst, const __fp16* restrict src, int M, int mask_stride) {
  for (int m = 0; m < M; ++m) {
    const __fp16* src_row = src + (size_t)m * mask_stride;
    float* dst_row = dst + (size_t)m * mask_stride;
    int n = 0;
    for (; n <= mask_stride - 64; n += 64) {
      HVX_Vector v_mask_hf = vmemu(src_row + n);
      v_mask_hf = Q6_Vh_vshuff_Vh(v_mask_hf);
      HVX_VectorPair v_mask_sf = Q6_Wsf_vcvt_Vhf(v_mask_hf);
      vmemu(dst_row + n) = Q6_V_lo_W(v_mask_sf);
      vmemu(dst_row + n + 32) = Q6_V_hi_W(v_mask_sf);
    }
    for (; n < mask_stride; ++n) {
      dst_row[n] = (float)src_row[n];
    }
  }
}

AEEResult htp_ops_flash_attn(uint8_t* pOut,
                             uint8_t* pQ,
                             uint8_t* pK,
                             uint8_t* pV,
                             uint8_t* pMask,
                             uint8_t* pWorkspace,
                             uint8_t* pPastK,
                             uint8_t* pPastV,
                             int32_t qo_len, int32_t seq_current,
                             int32_t seq_add, int32_t n_heads, int32_t n_kv_heads, int32_t head_dim, float scale, int32_t mask_stride,
                             int32_t max_kv_len, int32_t value_c4) {
  if (pK && pV && seq_add > 0) {
      htp_ops_push_kv(pPastK, pPastV, pK, pV, seq_current, seq_add, n_kv_heads, head_dim, max_kv_len,
                      value_c4, 0, seq_add);
  }
  if (flash_attn_try_single_token_output(pOut, pV, qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim,
                                         value_c4)) {
    return 0;
  }

  __fp16 *outBase = (__fp16 *)(pOut);
  const __fp16 *qBase = (const __fp16 *)(pQ);
  const __fp16 *maskBase = (const __fp16 *)(pMask);
  uint8_t *workspaceBase = (uint8_t *)(pWorkspace);
  __fp16 *pastKBase = (__fp16 *)(pPastK);
  __fp16 *pastVBase = (__fp16 *)(pPastV);

  int seq_len = seq_current + seq_add;
  int total_tasks = sync_attention_total_tasks(qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim, seq_len,
                                               mask_stride);
  int task_rows = sync_attention_task_rows(qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim, seq_len,
                                           mask_stride);
  size_t worker_workspace_bytes = sync_attention_head_workspace_bytes(task_rows, seq_current + seq_add);
  int worker_slots = sync_attention_pick_task_count(total_tasks);
  if (g_max_num_workers > 0 && worker_slots < (int)g_max_num_workers) {
    worker_slots = (int)g_max_num_workers;
  }
  float* maskFp32Base = NULL;
  if (maskBase != NULL && mask_stride > 0) {
    size_t maskFp32Offset = (size_t)worker_slots * worker_workspace_bytes;
    maskFp32Base = (float*)(workspaceBase + maskFp32Offset);
    preprocess_mask_to_fp32(maskFp32Base, maskBase, qo_len, mask_stride);
  }

  int ret = sync_attention(outBase, qBase, maskFp32Base, workspaceBase, pastKBase, pastVBase, qo_len, seq_current, seq_add,
                           n_heads, n_kv_heads, head_dim, scale, mask_stride);

  return ret;
}

AEEResult htp_ops_flash_attn_pages(uint8_t* pOut,
                                   uint8_t* pQ,
                                   uint8_t* pK,
                                   uint8_t* pV,
                                   uint8_t* pMask,
                                   uint8_t* pWorkspace,
                                   uint8_t** pPastKPages,
                                   uint8_t** pPastVPages,
                                   int32_t qo_len, int32_t seq_current,
                                   int32_t seq_add, int32_t n_heads, int32_t n_kv_heads, int32_t head_dim, float scale,
                                   int32_t mask_stride, int32_t max_kv_len, int32_t page_count, int32_t page_size,
                                   int32_t value_c4) {
  if (page_count <= 0 || page_size <= 0 || (page_size % 32) != 0) {
    return AEE_EBADPARM;
  }
  AsyncPushKVPagesState asyncPush = {};
  AsyncPushKVPagesState* asyncPushPtr = NULL;
  if (pK && pV && seq_add > 0) {
    if (seq_current < page_size) {
      const int kv_stride_bytes = n_kv_heads * head_dim * (int)sizeof(__fp16);
      int sync_push_len = seq_add;
      if (seq_current == 0 && mask_stride < 0 && qo_len > 1 && page_count == 1 &&
          seq_add > 64 && seq_add <= page_size) {
        sync_push_len = 64;
      }
      AEEResult ret = htp_ops_push_kv_pages(pPastKPages, pPastVPages, pK, pV, seq_current, sync_push_len,
                                            n_kv_heads, head_dim, page_count, page_size, value_c4, 0, seq_add);
      if (ret != 0) {
        return ret;
      }
      if (sync_push_len < seq_add) {
        asyncPush.done = 0;
        asyncPush.status = 0;
        asyncPush.pastKPages = pPastKPages;
        asyncPush.pastVPages = pPastVPages;
        asyncPush.K = pK + (size_t)sync_push_len * kv_stride_bytes;
        asyncPush.V = value_c4 ? pV : pV + (size_t)sync_push_len * kv_stride_bytes;
        asyncPush.seq_current = seq_current + sync_push_len;
        asyncPush.seq_add = seq_add - sync_push_len;
        asyncPush.n_kv_heads = n_kv_heads;
        asyncPush.head_dim = head_dim;
        asyncPush.page_count = page_count;
        asyncPush.page_size = page_size;
        asyncPush.value_c4 = value_c4;
        asyncPush.value_token_offset = sync_push_len;
        asyncPush.value_seq_len = seq_add;

        worker_pool_job_t pushJob;
        pushJob.fptr = push_kv_pages_async_worker;
        pushJob.dptr = &asyncPush;
        if (worker_pool_submit(NULL, pushJob) == 0) {
          asyncPushPtr = &asyncPush;
        } else {
          ret = htp_ops_push_kv_pages(pPastKPages, pPastVPages,
                                      pK + (size_t)sync_push_len * kv_stride_bytes,
                                      value_c4 ? pV : pV + (size_t)sync_push_len * kv_stride_bytes,
                                      seq_current + sync_push_len, seq_add - sync_push_len,
                                      n_kv_heads, head_dim, page_count, page_size,
                                      value_c4, sync_push_len, seq_add);
          if (ret != 0) {
            return ret;
          }
          asyncPush.done = 1;
        }
      }
      if (flash_attn_try_single_token_output(pOut, pV, qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim,
                                             value_c4)) {
        return 0;
      }
    } else {
      asyncPush.done = 0;
      asyncPush.status = 0;
      asyncPush.pastKPages = pPastKPages;
      asyncPush.pastVPages = pPastVPages;
      asyncPush.K = pK;
      asyncPush.V = pV;
      asyncPush.seq_current = seq_current;
      asyncPush.seq_add = seq_add;
      asyncPush.n_kv_heads = n_kv_heads;
      asyncPush.head_dim = head_dim;
      asyncPush.page_count = page_count;
      asyncPush.page_size = page_size;
      asyncPush.value_c4 = value_c4;
      asyncPush.value_token_offset = 0;
      asyncPush.value_seq_len = seq_add;

      worker_pool_job_t pushJob;
      pushJob.fptr = push_kv_pages_async_worker;
      pushJob.dptr = &asyncPush;
      if (worker_pool_submit(NULL, pushJob) == 0) {
        asyncPushPtr = &asyncPush;
      } else {
        AEEResult ret = htp_ops_push_kv_pages(pPastKPages, pPastVPages, pK, pV, seq_current, seq_add,
                                              n_kv_heads, head_dim, page_count, page_size,
                                              value_c4, 0, seq_add);
        if (ret != 0) {
          return ret;
        }
        asyncPush.done = 1;
      }
    }
  } else if (flash_attn_try_single_token_output(pOut, pV, qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim,
                                                value_c4)) {
    return 0;
  }

  __fp16 *outBase = (__fp16 *)(pOut);
  const __fp16 *qBase = (const __fp16 *)(pQ);
  const __fp16 *maskBase = (const __fp16 *)(pMask);
  uint8_t *workspaceBase = (uint8_t *)(pWorkspace);

  int seq_len = seq_current + seq_add;
  int total_tasks = sync_attention_total_tasks(qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim, seq_len,
                                               mask_stride);
  int task_rows = sync_attention_task_rows(qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim, seq_len,
                                           mask_stride);
  int allow_online_pages = 1;
  int online_block_size = page_size;
  int use_online_pages = sync_attention_use_online_pages(allow_online_pages, head_dim, page_size, seq_len, mask_stride);
  size_t worker_workspace_bytes = use_online_pages ?
      sync_attention_page_block_workspace_bytes(task_rows, online_block_size, head_dim) :
      sync_attention_page_head_workspace_bytes(task_rows, seq_len, head_dim);
  int worker_slots = sync_attention_pick_task_count(total_tasks);
  if (g_max_num_workers > 0 && worker_slots < (int)g_max_num_workers) {
    worker_slots = (int)g_max_num_workers;
  }
  float* maskFp32Base = NULL;
  const float* maskForAttention = NULL;
  if (maskBase != NULL && mask_stride > 0) {
    if (use_online_pages) {
      maskForAttention = (const float*)maskBase;
    } else {
      size_t maskFp32Offset = (size_t)worker_slots * worker_workspace_bytes;
      maskFp32Base = (float*)(workspaceBase + maskFp32Offset);
      preprocess_mask_to_fp32(maskFp32Base, maskBase, qo_len, mask_stride);
      maskForAttention = maskFp32Base;
    }
  }

  (void)max_kv_len;
  int ret = sync_attention_pages(outBase, qBase, maskForAttention, workspaceBase, pPastKPages, pPastVPages,
                                 qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim, scale, mask_stride,
                                 page_count, page_size, asyncPushPtr, allow_online_pages);
  if (asyncPushPtr != NULL) {
    while (!asyncPush.done) {
      asm volatile("pause(#8)" ::: "memory");
    }
    if (asyncPush.status != 0) {
      return asyncPush.status;
    }
  }
  return ret;
}
