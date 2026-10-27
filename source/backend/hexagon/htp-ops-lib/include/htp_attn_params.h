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

static inline size_t htp_attn_align128(size_t v) {
  return (v + 127u) & ~(size_t) 127u;
}

// VTCM the HMX kernels bump-allocate above the regions, taking the worse of
// attn_hmx_matmul_pages_{qk,sv}. SV dominates: its K is the KV extent rather than head_dim, so its
// double-buffered weight staging grows with seq_len -- 0.8 MB at seq 512, 2.5 MB at 2048, 5.0 MB at 4096.
//
// This was a flat 2 MB reserve, which seq 2048 already exceeded; it stayed safe only because the regions
// happened to leave more than that unused. vtcm_seq_alloc has no bounds check, so being wrong here
// silently corrupts the regions instead of faulting. Compute it rather than guess.
static inline size_t htp_attn_vtcm_kernel_bytes(int rows, int seq_len) {
  const size_t npc    = HTP_ATTN_HMX_KV_BLOCK_TILES;
  const size_t max_kp = (size_t) ((seq_len + 31) / 32);
  const size_t mpp    = (size_t) ((seq_len + 63) / 64);
  const size_t mp     = (size_t) ((rows + 31) / 32);
  const size_t weight = 2u * npc * 32u * max_kp * 32u * sizeof(short);
  const size_t raw    = mp * mpp * 32u * 64u * sizeof(short);
  const size_t act    = mp * max_kp * 32u * 32u * sizeof(short);
  const size_t output = 32u * 32u * npc * sizeof(short);
  return weight + raw + act + output + 256u /* column scales */ + 4096u /* alignment slack */;
}

// VTCM bytes one worker needs for both regions: the QK scores (fp16 on this path, since it only uses the
// lower half of the fp32-sized buffer) plus linear_S and temp_O.
static inline size_t htp_attn_vtcm_regions_bytes(int rows, int seq_len, int head_dim) {
  const size_t n_padded  = (size_t) ((seq_len + 31) / 32 * 32);
  const size_t hd_padded = (size_t) ((head_dim + 31) / 32 * 32);
  const size_t scores    = htp_attn_align128((size_t) rows * n_padded * sizeof(short));
  const size_t act       = htp_attn_align128((size_t) rows * n_padded * sizeof(short)) +
                           htp_attn_align128((size_t) rows * hd_padded * sizeof(short));
  return scores + act;
}

// Do both per-worker regions plus the kernel scratch fit in VTCM for this shape? The DSP's region
// placement and this picker must agree, so both go through here.
static inline int htp_attn_vtcm_fits(int rows, int seq_len, int head_dim, int workers, size_t vtcm_size) {
  const size_t regions = (size_t) workers * htp_attn_vtcm_regions_bytes(rows, seq_len, head_dim);
  const size_t kernel  = htp_attn_vtcm_kernel_bytes(rows, seq_len);
  return regions + kernel <= vtcm_size;
}

// Does the score region alone fit? This is the condition sync_attention_try_vtcm_workspace actually
// needs, since it places scores first and only adds the activation region if that still fits.
static inline int htp_attn_vtcm_scores_fit(int rows, int seq_len, int workers, size_t vtcm_size) {
  const size_t n_padded = (size_t) ((seq_len + 31) / 32 * 32);
  const size_t scores   = htp_attn_align128((size_t) rows * n_padded * sizeof(short));
  return (size_t) workers * scores + htp_attn_vtcm_kernel_bytes(rows, seq_len) <= vtcm_size;
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
                                               size_t vtcm_size) {
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
    if (htp_attn_vtcm_scores_fit(rows, seq_len, workers, vtcm_size)) {
      return g;
    }
  }
  return q_rows;
}

#endif  // MNN_HTP_ATTN_PARAMS_H
