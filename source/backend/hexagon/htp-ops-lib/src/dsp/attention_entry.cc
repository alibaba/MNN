#include <stdlib.h>

#include "attention_private.hpp"

extern "C" AEEResult htp_ops_vision_attention_fp16(uint8_t *pOut, const uint8_t *pQ, const uint8_t *pK,
                                                   const uint8_t *pV, const uint8_t *pMask, uint8_t *pWorkspace,
                                                   int batch, int tokens, int heads, int headDim, float scale,
                                                   int maskStride, int providedWorkspaceBytes) {
  if (pOut == NULL || pQ == NULL || pK == NULL || pV == NULL || batch <= 0 || tokens <= 0 || heads <= 0 ||
      headDim <= 0) {
    return AEE_EBADPARM;
  }
  const size_t workspaceBytes = (size_t) tokens * sizeof(float) + 127;
  if (pWorkspace == NULL || providedWorkspaceBytes <= 0 || (size_t) providedWorkspaceBytes < workspaceBytes) {
    return AEE_EBADPARM;
  }
  float        *scores      = (float *) (((uintptr_t) pWorkspace + 127) & ~(uintptr_t) 127);
  const __fp16 *query       = (const __fp16 *) pQ;
  const __fp16 *key         = (const __fp16 *) pK;
  const __fp16 *value       = (const __fp16 *) pV;
  const __fp16 *mask        = (const __fp16 *) pMask;
  __fp16       *output      = (__fp16 *) pOut;
  const size_t  tokenStride = (size_t) heads * headDim;
  for (int b = 0; b < batch; ++b) {
    for (int q = 0; q < tokens; ++q) {
      for (int h = 0; h < heads; ++h) {
        float         maxScore = -INFINITY;
        const __fp16 *qPtr     = query + ((size_t) b * tokens + q) * tokenStride + (size_t) h * headDim;
        for (int k = 0; k < tokens; ++k) {
          const __fp16 *kPtr  = key + ((size_t) b * tokens + k) * tokenStride + (size_t) h * headDim;
          float         score = 0.0f;
          for (int d = 0; d < headDim; ++d) {
            score += (float) qPtr[d] * (float) kPtr[d];
          }
          score *= scale;
          if (mask != NULL && maskStride > 0) {
            score += (float) mask[((size_t) b * tokens + q) * maskStride + k];
          }
          scores[k] = score;
          maxScore  = score > maxScore ? score : maxScore;
        }
        const float sum    = sync_attention_exp_and_sum(scores, tokens, maxScore);
        const float invSum = sum > 0.0f ? 1.0f / sum : 0.0f;
        __fp16     *outPtr = output + ((size_t) b * tokens + q) * tokenStride + (size_t) h * headDim;
        for (int d = 0; d < headDim; ++d) {
          float result = 0.0f;
          for (int k = 0; k < tokens; ++k) {
            const __fp16 *vPtr = value + ((size_t) b * tokens + k) * tokenStride + (size_t) h * headDim;
            result += scores[k] * invSum * (float) vPtr[d];
          }
          outPtr[d] = (__fp16) result;
        }
      }
    }
  }
  return AEE_SUCCESS;
}

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

extern "C" AEEResult htp_ops_vision_flash_attention_fp16(uint8_t *pOut, const uint8_t *pQ, const uint8_t *pK,
                                                         const uint8_t *pV, const uint8_t *pMask, uint8_t *pWorkspace,
                                                         int batch, int tokens, int heads, int headDim, float scale,
                                                         int maskStride, int providedWorkspaceBytes) {
  if (pOut == NULL || pQ == NULL || pK == NULL || pV == NULL || batch <= 0 || tokens <= 0 || heads <= 0 ||
      headDim <= 0 || (headDim % 64) != 0 || (pMask != NULL && maskStride < tokens)) {
    return AEE_EBADPARM;
  }
  const int    seqBlocks    = (tokens + ATTN_HMX_KV_BLOCK - 1) / ATTN_HMX_KV_BLOCK;
  const int    kIcP         = (headDim + 31) / 32;
  const int    vOcP         = (headDim + 31) / 32;
  const size_t packedKBytes = (size_t) seqBlocks * heads * ATTN_HMX_KV_BLOCK_TILES * kIcP * 1024 * sizeof(__fp16);
  const size_t packedVBytes = (size_t) seqBlocks * heads * vOcP * ATTN_HMX_KV_BLOCK_TILES * 1024 * sizeof(__fp16);
  const int    totalTasks   = sync_attention_total_tasks(tokens, 0, tokens, heads, heads, headDim, tokens, maskStride);
  int          workerSlots  = sync_attention_pick_task_count(totalTasks);
  if (g_max_num_workers > 0 && workerSlots < (int) g_max_num_workers) {
    workerSlots = (int) g_max_num_workers;
  }
  const int    queryBlock  = tokens < 64 ? tokens : 64;
  const size_t workerBytes = sync_attention_head_workspace_bytes(queryBlock, tokens);

  size_t packedKOffset      = 0;
  size_t packedVOffset      = attn_align_128(packedKOffset + packedKBytes);
  size_t packedOutputOffset = attn_align_128(packedVOffset + packedVBytes);
  size_t attentionWorkspaceOffset =
    attn_align_128(packedOutputOffset + (size_t) queryBlock * heads * headDim * sizeof(__fp16));
  size_t maskOffset     = attn_align_128(attentionWorkspaceOffset + (size_t) workerSlots * workerBytes);
  size_t workspaceBytes = maskOffset;
  if (pMask != NULL) {
    workspaceBytes = attn_align_128(maskOffset + (size_t) queryBlock * maskStride * sizeof(float));
  }
  const size_t requiredWorkspaceBytes = workspaceBytes + 127;
  if (pWorkspace != NULL && (providedWorkspaceBytes <= 0 || (size_t) providedWorkspaceBytes < requiredWorkspaceBytes)) {
    return AEE_EBADPARM;
  }
  uint8_t *ownedWorkspace = NULL;
  if (pWorkspace == NULL) {
    ownedWorkspace = (uint8_t *) malloc(workspaceBytes + 127);
    pWorkspace     = ownedWorkspace;
  }
  if (pWorkspace != NULL) {
    pWorkspace = (uint8_t *) (((uintptr_t) pWorkspace + 127) & ~(uintptr_t) 127);
  }
  if (pWorkspace == NULL) {
    return AEE_ENOMEMORY;
  }
  uint8_t *packedK            = pWorkspace + packedKOffset;
  uint8_t *packedV            = pWorkspace + packedVOffset;
  __fp16  *packedOutput       = (__fp16 *) (pWorkspace + packedOutputOffset);
  uint8_t *attentionWorkspace = pWorkspace + attentionWorkspaceOffset;
  float   *maskFp32           = pMask != NULL ? (float *) (pWorkspace + maskOffset) : NULL;

  const size_t tensorStride    = (size_t) tokens * heads * headDim;
  const size_t maskStrideBatch = (size_t) tokens * maskStride;
  for (int b = 0; b < batch; ++b) {
    const __fp16 *query  = (const __fp16 *) pQ + b * tensorStride;
    const __fp16 *key    = (const __fp16 *) pK + b * tensorStride;
    const __fp16 *value  = (const __fp16 *) pV + b * tensorStride;
    __fp16       *output = (__fp16 *) pOut + b * tensorStride;
    const __fp16 *mask   = pMask != NULL ? (const __fp16 *) pMask + b * maskStrideBatch : NULL;

    memset(packedK, 0, packedKBytes);
    memset(packedV, 0, packedVBytes);
    AEEResult ret = htp_ops_push_kv(packedK, packedV, (uint8_t *) key, (uint8_t *) value, 0, tokens, heads, headDim,
                                    seqBlocks * ATTN_HMX_KV_BLOCK, 0, 0, tokens);
    if (ret != AEE_SUCCESS) {
      free(ownedWorkspace);
      return ret;
    }
    for (int qBase = 0; qBase < tokens; qBase += queryBlock) {
      const int queryRows = tokens - qBase < queryBlock ? tokens - qBase : queryBlock;
      if (mask != NULL) {
        preprocess_mask_to_fp32(maskFp32, mask + (size_t) qBase * maskStride, queryRows, maskStride);
      }
      ret = sync_attention(packedOutput, query + (size_t) qBase * heads * headDim, maskFp32, attentionWorkspace,
                           (__fp16 *) packedK, (__fp16 *) packedV, queryRows, tokens - queryRows, queryRows, heads,
                           heads, headDim, scale, mask != NULL ? maskStride : 0);
      if (ret != AEE_SUCCESS) {
        free(ownedWorkspace);
        return ret;
      }
      for (int h = 0; h < heads; ++h) {
        for (int pack = 0; pack < headDim / 64; ++pack) {
          const __fp16 *src = packedOutput + ((size_t) h * (headDim / 64) + pack) * queryRows * 64;
          for (int q = 0; q < queryRows; ++q) {
            __fp16 *dst = output + ((size_t) (qBase + q) * heads + h) * headDim + pack * 64;
            vmemu(dst)  = vmemu(src + (size_t) q * 64);
          }
        }
      }
    }
  }
  free(ownedWorkspace);
  return AEE_SUCCESS;
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
