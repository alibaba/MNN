#ifndef MNN_HTP_ATTN_PARAMS_H
#define MNN_HTP_ATTN_PARAMS_H

// Grouped-causal prefill shaping, shared by the host (DDR workspace sizing in
// execution/HexagonAttention.cpp) and the DSP (task shaping in src/dsp/attention_sync_*.cc).
//
// These used to be two manually-synced constants (MNN_ATTENTION_CAUSAL_GROUP_Q_ROWS and
// HEXAGON_ATTN_CAUSAL_GROUP_Q_ROWS) whose comments told each other to stay equal. Having one definition
// instead is the point: the host now only needs the *upper bound* to size its buffer, while the DSP picks
// the best value for the actual shape, so the two can no longer disagree in a way that corrupts memory.
//
// Host-includable on purpose: plain C, no HVX types, no DSP headers.

#include <stddef.h>

// Largest Q-row group. The host sizes its workspace for this, so the DSP must never exceed it.
#define HTP_ATTN_CAUSAL_GROUP_Q_ROWS_MAX 64

// KV tiles per weight chunk, mirroring ATTN_HMX_KV_BLOCK_TILES on the DSP side.
#define HTP_ATTN_HMX_KV_BLOCK_TILES 8

// KV pages the all-page SV fusion holds resident, mirroring ATTN_HMX_MAX_FUSED_PAGES.
#define HTP_ATTN_HMX_MAX_FUSED_PAGES 8

static inline size_t htp_attn_align128(size_t v) {
  return (v + 127u) & ~(size_t) 127u;
}

// VTCM the HMX kernels bump-allocate above the regions: the worse of the non-fused page loop and the
// all-page fusion.
//
// `kv_chunk` is the KV extent one kernel invocation walks. The paged kernels stage one page at a time
// (page_size, 256), while the non-paged attn_hmx_matmul takes the whole valid_end. Sizing this by
// seq_len unconditionally -- which this did -- over-estimates the paged case (the one the LLM uses) by
// 4..10x: at seq 4096 the estimate was 12.6 MB for rows=512 against ~3.3 MB actually allocated, and it
// made the group picker shrink groups for nothing (gqa 8 / seq 4096 picked 16 where 64 fits).
// vtcm_seq_alloc has no bounds check, so this must stay an upper bound -- hence the max over branches
// and over QK's head_dim-sized K and SV's KV-sized K.
static inline size_t htp_attn_vtcm_kernel_bytes(int rows, int head_dim, int kv_chunk) {
  const size_t npc    = HTP_ATTN_HMX_KV_BLOCK_TILES;
  const size_t fused    = HTP_ATTN_HMX_MAX_FUSED_PAGES;
  const size_t mp     = (size_t) ((rows + 31) / 32);
  const size_t hd_kp    = (size_t) ((head_dim + 31) / 32);
  const size_t hd_pp    = (size_t) ((head_dim + 63) / 64);
  const size_t kv_kp    = (size_t) ((kv_chunk + 31) / 32);
  const size_t kv_pp    = (size_t) ((kv_chunk + 63) / 64);
  const size_t kp       = hd_kp > kv_kp ? hd_kp : kv_kp;
  const size_t pp       = hd_pp > kv_pp ? hd_pp : kv_pp;
  const size_t output = 32u * 32u * npc * sizeof(short);
  // Non-fused: double-buffered weight over npc chunks, raw/act per M-tile.
  const size_t nonfused = 2u * npc * 32u * kp * 32u * sizeof(short) + mp * pp * 32u * 64u * sizeof(short) +
                          mp * kp * 32u * 32u * sizeof(short) + output;
  // All-page fusion: one weight set per resident page (its np is head_dim / 32), activations per
  // (page, M-tile). The tile path skips the raw/act staging, so this covers its worst case.
  const size_t fused_bytes = fused * hd_kp * 32u * kv_kp * 32u * sizeof(short) +
                             mp * kv_pp * 32u * 64u * sizeof(short) + fused * mp * kv_kp * 32u * 32u * sizeof(short) +
                             output;
  const size_t kernel = nonfused > fused_bytes ? nonfused : fused_bytes;
  return kernel + 256u /* column scales */ + 4096u /* alignment slack */;
}

// Does the score region alone fit? This is the condition sync_attention_try_vtcm_workspace actually
// needs, since it places scores first and only adds the activation region if that still fits.
static inline int htp_attn_vtcm_scores_fit(int rows, int seq_len, int head_dim, int kv_chunk, int workers,
                                           size_t vtcm_size) {
  const size_t n_padded = (size_t) ((seq_len + 31) / 32 * 32);
  const size_t scores   = htp_attn_align128((size_t) rows * n_padded * sizeof(short));
  return (size_t) workers * scores + htp_attn_vtcm_kernel_bytes(rows, head_dim, kv_chunk) <= vtcm_size;
}

// Q rows per grouped-causal prefill block.
//
// Bigger is better for two reasons. K/V traffic: a block reads K/V once and reuses it across all
// gqa_factor*rows M-tiles, so total K/V DMA is ~qo_len^2/(2*group_q_rows). And submission count: each
// block costs one QK and one SV round trip through the single-threaded hmx_queue, and with 8 workers
// contending for it that queueing -- not the queue's own work -- is what sets attention's wall clock.
//
// So: the largest group whose score region fits. An earlier version required BOTH regions to fit, which
// at seq 2048 picked 32 instead of 64 and measured 3.4% faster at the time (the activation region being
// resident was worth +3.3% at pp512 and +6.2% at pp1024). That reversed once the softmax went fp16: 64
// then measured +2.7% at pp2048, because halving the submission count outweighs keeping the activation
// region in VTCM. Shorter sequences are unaffected -- 64 fits both ways there.
static inline int htp_attn_causal_group_q_rows(int qo_len, int seq_len, int head_dim, int gqa_factor, int workers,
                                               size_t vtcm_size, int kv_chunk) {
  int q_rows = HTP_ATTN_CAUSAL_GROUP_Q_ROWS_MAX;
  if (qo_len < q_rows) {
    q_rows = qo_len;
  }
  if (q_rows < 1) {
    q_rows = 1;
  }
  // Only the full-size group is worth shrinking; below it the K/V re-read already dominates.
  if (q_rows < HTP_ATTN_CAUSAL_GROUP_Q_ROWS_MAX || gqa_factor < 1) {
    return q_rows;
  }
  if (workers < 1) {
    workers = 1;
  }
  for (int g = q_rows; g >= 16; g >>= 1) {
    const int rows = gqa_factor * g;
    if (htp_attn_vtcm_scores_fit(rows, seq_len, head_dim, kv_chunk, workers, vtcm_size)) {
      return g;
    }
  }
  return q_rows;
}

#endif  // MNN_HTP_ATTN_PARAMS_H
