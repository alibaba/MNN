//
//  MetalEnv.hpp
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalEnv_hpp
#define MetalEnv_hpp

#import "core/Macro.h"
#if MNN_METAL_ENABLED

#include <cstdlib>
#include <cstring>

namespace MNN {

// Single registry for every MNN Metal environment switch.
// Values are parsed once on first use and never change afterwards.
//
// RULES for adding a switch:
//   1. Add a field here (no scattered getenv calls in backend code).
//   2. Document it in skills/metal-optimize/env-registry.md.
//   3. Prefer MNN_METAL_DISABLE_* for default-on and MNN_METAL_ENABLE_*
//      for default-off semantics; use named strings for multi-value modes.
struct MetalEnv {
    // ---- performance path ----
    // MNN_ENABLE_FLASH_ATTN_PREFILL: -1 unset (config decides) / 0 force off / 1 force on.
    int flashAttnPrefill;
    // MNN_METAL_PREFILL_INSHADER_DEQUANT_SGMATRIX: 0 unset (size threshold
    // decides) / 1 force in-shader / -1 force outer-dequant. Only consulted on
    // non-tensor-API devices (M5+ always takes outer-dequant + tensor API).
    int prefillInshaderDequant;
    // MNN_METAL_RESIZE_WAIT: 0 "local" per-backend fence (default) /
    // 1 "global" legacy drain / 2 "none" skip both (experiment only).
    int resizeWaitMode;
    // MNN_METAL_PREFILL_FA_TENSORAPI: fused prefill attention on the Metal
    // tensor API (prefill_flash_attn_tc). Keeps S and O in registers for the
    // whole KV sweep so the O(n^2) score matrix never reaches global memory,
    // unlike the three-stage path.
    //  -1 (unset) = default resolved from the data-driven mCausalLayout flag
    //       (standard causal mask => on, arbitrary mask => off). Arbitrary masks
    //       can never enable it since this kernel hard-codes causal.
    //   1 = explicit on (still gated to causal layout in MetalAttention)
    //   0 = explicit off, legacy prefill paths untouched
    // Further gated by isSupportTensorCoopInput() (M5+ only), so on
    // M4/M3/iPhone the default is a no-op.
    int prefillFaTensorApi;
    // MNN_METAL_PREFILL_FA_SG: M4 fused prefill (prefill_flash_attn_sg).
    // STEEL-like tiles, S/O in simdgroup fragments. -1 unset = auto-on at
    // seq>=512 only for the verified M4 Pro 32q/8kv/head_dim128 shape, and
    // seq>=1024 on other eligible non-tensor-coop shapes; 1 = force on,
    // 0 = force off.
    int prefillFaSg;
    // MNN_METAL_FASG_Q_REG=0: rollback switch. Default keeps the HEAD_DIM/8 Q
    // fragments in registers instead of re-issuing them from smem on every kv
    // tile (the simdgroup-matrix analogue of the tc kernel's Q_REG).
    bool fasgQReg;
    // MNN_METAL_FASG_NSG8=0: rollback switch. Default runs 8 simdgroups (256
    // threads) instead of 4, so the q tile is 64 rows at unchanged per-lane
    // register pressure. Halves how many times each layer re-reads the device
    // K/V stream. Only pays once LOADBATCH has removed the fragment-load stalls
    // that used to mask it.
    bool fasgNsg8;
    // MNN_METAL_FASG_LOADBATCH=N: issue N smem fragment loads before any MMA
    // consumes one, so load latency is covered by the preceding MMAs instead of
    // stalling the MMA that needs the fragment. N is the V-fragment batch width;
    // 0 disables. The batch loops must stay fully unrolled: a dynamically
    // indexed simdgroup-matrix array spills out of registers and costs ~2.8x.
    int fasgLoadBatch;
    // MNN_METAL_DISABLE_FATC_Q_REG=1: rollback switch for the Q-in-registers
    // form of prefill_flash_attn_tc. Default (unset) keeps Q cached in
    // registers: the q-tile is loaded and pre-scaled once instead of being
    // re-gathered from device memory on every (kv tile, head_dim frag) pair,
    // which is O(kv_lim/BK) redundant loads at long prefill. Only reachable on
    // tensor-coop devices (M5+).
    bool faTcQRegDisabled;
    // MNN_METAL_FATC_QK_K32: widen the tc QK matmul K dimension 16 -> 32
    // (matmul2d_descriptor(16,32,32)). Halves the head_dim accumulate calls
    // (8 -> 4 at HEAD_DIM=128), doubling per-call tensor work. A/B keep the
    // K=16 per-lane mapping and add one +16 K-axis level; the destination
    // layout is identical, so the softmax/PV handoff is unchanged. Only
    // reachable on tensor-coop devices (M5+). Tri-state: unset = on, 0 = force
    // K=16, 1 = force K=32.
    bool faTcQkK32;
    // MNN_METAL_FATC_O_CT: keep the online-softmax O accumulator as a
    // persistent cooperative_tensor destination. Skips the per-iteration copy
    // between mm_pv's destination CT and a scalar float array, and keeps O
    // values register-resident across the whole kv-loop. Only reachable on
    // tensor-coop devices (M5+).
    // Tri-state: unset = on, 0 = force off, 1 = force on.
    bool faTcOCt;
    // MNN_METAL_FATC_KV_DEV_TENSOR (default ON, set 0 to disable): hand K/V to
    // matmul2d as `tensor` handle slices pointing straight at device memory
    // (strided tensor_inline over the kv cache) instead of hand-filling the
    // right input cooperative tensor. The matmul then issues its own operand
    // loads, so the per-lane B fill -- BK*HEAD_DIM/32 = 128 half moves per lane
    // per kv tile for K plus another 128 for V, the dominant scalar cost and one
    // no BK/K widening can reduce -- disappears, with no staging, no barrier,
    // and no kv loop widened to the threadgroup max q row.
    // Staging K/V through threadgroup smem first was falsified twice: once still
    // hand-packing every operand out of smem, once as a staged tensor-handle form
    // that dropped the packing but kept the stage + barrier. Requires faTcOCt.
    bool faTcKvDevTensor;
    // MNN_METAL_FATC_QREV: reverse the q-tile -> threadgroup.x mapping. Under
    // causal masking a threadgroup's kv range is q_row_base+16 long, so cost
    // grows with the q-tile index, while the grid issues x-major: the cheapest
    // tiles start first and the longest ones last, so the dispatch ends with a
    // few long threadgroups running on a mostly idle GPU. Reversing issues the
    // longest first and leaves cheap tiles to fill the tail. Pure index
    // permutation - no extra registers, no extra traffic. Neutral below
    // seq2048, where the dispatch is short enough that the tail is not the
    // limiter. Default on; set 0 to restore the x-major mapping.
    bool faTcQRev;
    // MNN_METAL_FATC_Q_CT: keep the QK left-input cooperative tensors alive
    // across the whole kv loop instead of building and filling them once per kv
    // tile. Q is loop-invariant, yet faTcQReg only got it as far as a register
    // array: every tile still repacks q_reg into a fresh input CT, which is
    // FATC_TDK * (FATC_QK_K / 2) scalar half stores per lane per tile (64 at
    // HEAD_DIM=128 with QK_K=32). Same lever as faTcOCt on the destination
    // side. The CTs replace q_reg rather than adding to it - both hold exactly
    // HEAD_DIM/2 halves per lane - so the register footprint should not move.
    // Requires faTcQkK32 (the shader only spells out the persistent-CT fill for
    // FATC_TDK <= 4). Tri-state: unset = on, 0 restores the per-tile refill.
    bool faTcQCt;
    // MNN_METAL_FATC_PV_K32: widen the PV matmul2d K from 16 to 32 so one call
    // consumes the whole BK=32 kv tile, halving the PV call count (8 -> 4 at
    // HEAD_DIM=128). This is the PV-side counterpart of faTcQkK32; the device V
    // byte count is unchanged, only the per-call dispatch overhead drops. In the
    // K=32 operand layout the left operand is exactly s_acc[0..15], and the right
    // operand adds one +16 K level at bit 4 of the element index. Requires faTcOCt
    // (only the persistent-CT PV block implements the wide-K fill).
    // Tri-state: unset = on, 0 = force K=16.
    bool faTcPvK32;
    // MNN_METAL_FATC_DSPLIT: how many head_dim slices the tc prefill kernel
    // spreads over the dispatch grid's z dimension. The persistent O accumulator
    // is head_dim/32 destination cooperative tensors of 16 floats each, so at
    // head_dim 256 it alone occupies 128 float registers per lane and the kernel
    // collapses under register pressure. Splitting z-ways divides the O footprint
    // by DSPLIT at the cost of recomputing QK per slice (1.5x FLOPs at DSPLIT=2),
    // which is the only register-relief option here that needs no barrier --
    // every staged variant tried was slower.
    // unset = -1 (auto: 2 at head_dim 256, 1 otherwise); N = explicit slices.
    int faTcDSplit;
    // MNN_METAL_ATTN_QSPLIT / MNN_METAL_ATTN_QSPLIT_MB: the three-stage prefill
    // path materializes mTempQK and mTempSoftMax, together 2 * B * H * seq *
    // kv_max halves -- 16 GiB at 16 q heads and seq = kv = 16384, which thrashes
    // a 16 GB machine. Splitting the q sequence into N pieces divides that by N
    // at no arithmetic cost, since each piece is an independent set of q rows.
    // QSPLIT forces N (powers of two only; mainly to exercise N>1 in tests),
    // QSPLIT_MB is the per-piece scratch budget the auto policy sizes N against.
    // unset = 0 (derive N from the budget); N = explicit piece count.
    int attnQSplit;
    // unset = 128 MB. Below the split point the pieces stop filling the GPU;
    // above it the unsplit form is latency-bound.
    int attnQSplitMB;
    // MNN_METAL_LN_TOKENS_PER_TG: how many tokens share one threadgroup in the
    // C4 RMSNorm at prefill. The reduction stays inside a simdgroup either way,
    // so this only trades threadgroup count against threadgroup width.
    // unset = 8 (auto-on at outside>=256), 1 = one threadgroup per token.
    int lnRowParallel;
    // MNN_METAL_LN_SUM_REREAD: in the binary (add + RMSNorm) C4 kernels the
    // normalize pass needs the summed row a second time. Re-adding in0+in1 costs
    // two DRAM reads; reading back the sum the thread just wrote to out0 costs
    // one, cutting the kernel's traffic.
    // unset = 1 (on), 0 = re-add both inputs.
    int lnSumReread;
    // MNN_METAL_LN_TOKEN_LANE: map the simdgroup lane to the token instead of the
    // channel in the binary (add + RMSNorm) C4 prefill kernel. The C4 activation
    // is channel-major (idx = c * batch + token), so lane-per-channel makes a
    // 32-lane load touch 32 addresses batch*16 bytes apart, while lane-per-token
    // makes it 32 contiguous float4; the row reduction also becomes lane-private,
    // dropping simd_sum. Needs outside >= 32 to fill a simdgroup, so prefill only.
    // Only the very longest prefill wins: the channel form holds bandwidth to
    // moderate batch and then degrades, while the token form scales cleanly but
    // is latency-bound at small batch because one lane walks the whole row.
    // Kept as a probe for devices where the channel form degrades earlier.
    // unset = 0 (off), 1 = token-parallel.
    int lnTokenLane;
    // MNN_METAL_DECODE_SDPA: single-pass fused decode attention.
    // decode_splitkv with ntg pinned to 1, no reduce dispatch, final
    // output written by the kernel itself (one threadgroup produces the whole
    // normalized row). It replaces the split-KV path on the same
    // eligibility/threshold, so it defaults ON (auto) just like split-KV, which
    // is already default-on for all devices.
    // unset = 1 (auto kv threshold, same clamped threshold as split-KV);
    // 0 = explicit off (legacy split-KV path); N>1 = explicit kv threshold
    // override (probe). See decodeSdpaNsg for the device tier.
    int decodeSdpa;
    // MNN_METAL_DECODE_SDPA_NSG: simdgroups per threadgroup for the single-pass
    // kernel. Allowed {4, 8, 16, 32}. Default 0 = device-tiered (resolved in
    // MetalAttention): tensor-API/M5 -> 32; non-tensor-API/M4-class -> 16.
    // M1/M2/M3/iPhone uncalibrated, inherit the M4 branch.
    int decodeSdpaNsg;
    // MNN_METAL_DECODE_SDPA_QH_PER_TG: q heads per threadgroup for the
    // single-pass kernel (SDPA_QH_PER_TG). Allowed {1, 2, 4, 8}; must divide the
    // GQA group_size or it is ignored. 1 = one TG per q head, each q head
    // re-reading the shared KV rows; >1 fetches each K/V row once for that many
    // q heads, trading threadgroup count for KV read requests. Default 0 = auto
    // (resolved in MetalAttention by group_size).
    int decodeSdpaQhPerTg;
    // MNN_METAL_DECODE_SDPA_NTG: kv threadgroups for the 2-pass split-KV decode
    // SDPA. N>1 (power of two, <=64) splits the kv sweep over N threadgroups on
    // grid.x, each writing (S, m, unnormalized O) partials that a second
    // one-simdgroup-per-row kernel recombines. Widens the dispatch and shortens
    // the per-threadgroup dependency chain at long ctx, at the cost of a partial
    // round-trip through device memory. Fixed by env or by the resize-time auto
    // rule (never kv-derived at replay time) so the recorded encode-replay grids
    // stay kv-independent.
    // unset = 0 (auto, resolved in MetalAttention); 1 = explicit single pass.
    // The auto band is narrow: only long-context, wide-q-head shapes gain from
    // splitting the kv sweep. See decodeSdpaNsg: the auto band also lifts nsg
    // 8 -> 16.
    int decodeSdpaNtg;
    // ---- fusion / misc ----
    // MNN_METAL_DISABLE_LN_FUSION=1: keep AddRMSNorm and the fused projections
    // as two dispatches. Default false keeps AddRMSNorm folded into the GEMV.
    bool lnFusionDisabled;
    // MNN_METAL_DISABLE_GATE_UP_FUSION=1: disable Gate/Up leader/follower fusion.
    bool gateUpFusionDisabled;
    // MNN_METAL_DISABLE_QKV_FUSION=1: disable Q/K/V leader/follower fusion.
    bool qkvFusionDisabled;
    // MNN_METAL_GEMV_W16=0: drop the Q4 GEMV_QBLOCK_W16 specialization (16-byte
    // uint4 weight reads, compile-time quads-per-block, per-lane contiguous pair
    // runs) and fall back to the generic ushort4 body. One gate in
    // MetalConvolution1x1::onResize covers every consumer: the plain and split-K
    // 2sg GEMV and the QKV / GateUp / LN fused pipelines. The g16 lm_head kernel
    // has no W16 body -- it is the one consumer where W16 measures as a loss.
    // Note for A/B work: the wide shape requires the W16 layout, so turning this
    // off also drops the fused QKV back to the narrow split.
    bool gemvW16Disabled;
    // MNN_METAL_LN_STAGE_FP32=1: keep the LN_STAGE staging vector float4 instead
    // of ftype4 (half4 in fp16 mode). The half staging halves the threadgroup
    // footprint and matches the unfused AddRMSNorm fp16 output precision; this
    // is the rollback + A/B switch.
    bool lnStageFp32;
    // MNN_METAL_LN_STAGE: tri-state override for the staged LN prologue.
    // unset = auto (on only where staging measured net positive, i.e.
    // hidden <= 1024; see MetalConvolution1x1::setupLNFusion), 1 = force on up
    // to the threadgroup budget, 0 = force off. The old boolean env was removed
    // on 2026-09-02 because the auto rule then coincided with the memory budget
    // and the env could not express a distinct arm; the narrower auto rule
    // brings both arms back.
    int lnStage;
    // MNN_METAL_QKV_MERGE: tri-state override for laying the fused group's member
    // weights and dequant scales end to end in one allocation instead of binding
    // one weight + one scale stream per member (see
    // MetalConvolution1x1::setupQKVFusion). unset = auto (3-member W4 groups
    // only), 1 = also merge 4-member groups, 0 = never merge. The member count is
    // the number of concurrent DRAM streams the dispatch opens, so this is the
    // knob that separates "fused shader body" from "N concurrent streams" on a
    // byte-matched arm.
    int qkvMerge;
    // MNN_METAL_QKV_MERGED_OUT: tri-state override for laying the fused group's
    // member output tensors into one allocation, which lets the shader address
    // every projection as one bound pointer plus a per-projection offset instead
    // of selecting one of N bound pointers (see
    // MetalConvolution1x1::setupQKVFusion). unset = auto (on), 1 = force on,
    // 0 = force off, which falls back to per-member static outputs and the
    // branch-selected pointer. The win only shows up on arms whose weight
    // footprint is past the system-level-cache cliff; below it the two forms
    // measure the same. The mechanism is not established -- this knob prices the
    // difference, it does not explain it.
    int qkvMergedOut;
    // MNN_METAL_QKV_PACKED_GRID: tri-state override for the fused group's grid
    // shape. unset = auto (packed only when the members differ in outputChannel,
    // where it avoids launching threadgroups that just early-return), 1 = always
    // packed, 0 = always rectangular. Both forms are correct on any group: the
    // packed bases in qkv_seg are populated unconditionally. Packed makes the
    // dispatch a flat 1D grid and rectangular makes it (x, y, numProj), so this
    // is the knob that separates grid dimensionality from the shader body.
    int qkvPackedGrid;
    // MNN_METAL_DISABLE_ROPE_X_CACHE=1: make the fused RoPE kernel re-read q/k from
    // device memory for the rotate pass instead of keeping the values the lane
    // already loaded for the RMSNorm sum in registers. Only the simdgroup path with
    // ropeHalfD % 32 == 0 can cache (the (i, i+ropeHalfD) pair then lands in the
    // same lane), so this is a rollback + A/B switch, not a shape gate.
    bool ropeXCacheDisabled;
    // MNN_METAL_ROPE_TILE: tri-state override for the token-tiled fused RoPE
    // kernel, where one simdgroup takes 8 consecutive tokens so the C4 reads
    // become one run per plane group instead of one request per lane. unset =
    // auto (on above the launch-bound floor, see kRopeTileMinSeq), 1 = force on
    // wherever the shape qualifies, 0 = never build it.
    int ropeTile;
    // MNN_METAL_ROPE_TILE_T: tokens per simdgroup in the tiled fused RoPE kernel.
    // 0/unset = default; a value that divides 32 and splits the head's channel
    // quads evenly overrides it. This is the run width of the read, and also sets
    // the staging footprint, so widening it trades request count for occupancy.
    int ropeTileT;
    // MNN_METAL_GEMV_W16_MID: force the W16 decode-GEMV lanes-per-block (how many
    // lanes share one quant block) instead of chooseQ4W16LanesPerBlock's pick. 0/unset =
    // auto; 1/2/4/8 = forced. chooseQ4W16LanesPerBlock only counts loop iterations, so
    // several candidates can tie while differing in how well lanes coalesce
    // (1 lane per block gives that lane a whole 128B block, i.e. 128B stride between lanes).
    int gemvW16LanesPerBlock;
    // MNN_METAL_GEMV_W16_DS=0: keep the gate/up silu dual-stream body on the
    // generic ushort4 weight reads instead of the W16 uint4 specialization
    // (rollback + A/B switch for the GEMV_QBLOCK_W16_DS branch).
    bool gemvW16DsDisabled;
    // MNN_METAL_GATEUP_SPLITK: override the gate/up silu dual-stream K split
    // factor (auto = 4 when the shape qualifies, else 0). 0 disables the split,
    // 1/2/4/8 force it; values the shape cannot divide evenly are ignored.
    int gateUpSplitK;
    // MNN_METAL_GATEUP_LN_STAGE=1: extend the staged LN prologue to the gate/up
    // LN-fused pipeline (off by default; the dual-stream body re-reads each
    // input quad once per threadgroup, which alone does not pay for staging).
    bool gateUpLnStage;
    // MNN_METAL_QKV_SPLITK_MAX_BLOCKS: quant-block ceiling for the narrow split. Split-K
    // buys parallelism with a threadgroup-memory reduction plus a barrier, so it
    // only pays when the unsplit dispatch is parallelism-starved. The block count
    // K/blocksize is what each simdgroup walks, so it is the direct measure of
    // that. The curve is not monotonic, so do not extrapolate to unmeasured
    // shapes; raise this to re-measure one instead.
    int qkvSplitKMaxBlocks;
    // MNN_METAL_ENABLE_LMHEAD_SPLITK: lm_head decode GEMV route override.
    // Unset (-1) selects the route from blockCount and oc; 0 uses legacy g16,
    // 1 routes through the g4 kernel, and 2 enables split-K inside g16.
    int lmheadSplitK;
    // MNN_METAL_W4W8_OUTER_DEQUANT_GEMM_TENSORAPI=1: take the outer-dequant +
    // fp GEMM path instead of the fused Q4/Q8 GEMM that unpacks weights
    // in-kernel (A/B baseline + emergency rollback). Only meaningful on
    // tensor-API devices (M5+), where the fused path is the default.
    bool w4w8OuterDequantGemm;
    // MNN_METAL_FUSED_Q4_KSPLIT: K-split x4 for fused-Q4 GEMM on speculative-block shapes.
    // Env unset = auto gate, "1" = force on, "0" = force off; stored as 0 / 1 / -1 respectively.
    int fusedQ4Ksplit;
    // MNN_METAL_FUSED_Q4_M8: M8 tile for the small-M shapes the K-split gate skips.
    // Env unset = auto, "1" = same as unset (tile only correct for area <= 8), "0" = force off;
    // stored as 0 / 1 / -1 respectively.
    int fusedQ4M8;
    // MNN_METAL_FUSED_Q4_KSPLIT_M8: stack the M8 tile on K-split.
    // Env unset = auto (area <= 8 gate), "1" = same as unset, "0" = force off (keep the M32 tile);
    // stored as 0 / 1 / -1 respectively.
    int fusedQ4KsplitM8;
    // MNN_METAL_FUSED_Q4_M64: M64 tile for fused-Q4 GEMM prefill.
    // Env unset = auto (on when area >= 64), "1" = force on, "0" = force off
    // (fall back to the M32 tile). Stored as 0 / 1 / -1 respectively.
    // Added for A/B validation on small models where the M64 tile may not
    // amortize its threadgroup-memory cost.
    // Widening the K tile and making N the fast-varying grid dim were both tried
    // and both lost: neither changes the staged bytes per flop, which is what
    // this kernel is bound by.
    int fusedQ4M64;
    // MNN_METAL_FUSED_Q4_SMEM_PAD=0 disables padding the fused-Q4 M64 GEMM's
    // staged operand row stride 32 -> 40 ftype. The unpadded activation store
    // address is (4*ml + r) * 8 + kl in ftype4 units, whose bank index does not
    // depend on ml, so all 16 ml lanes of a simdgroup collide. Default on. The
    // pad must keep the stride a multiple of 8 ftype so operand rows stay
    // 16-byte aligned for matmul2d; 36 does not, and produced wrong results at
    // fp16.
    int fusedQ4SmemPad;
    // MNN_METAL_FUSED_Q4_UNORM=0 falls back to the scalar shift/mask nibble
    // unpack in the fused-Q4 GEMMs. Default on.
    int fusedQ4Unorm;
    // MNN_METAL_FUSED_Q4_DOUBLE_BUF=0 restores single-buffered operand staging
    // in the fused-Q4 M64 prefill GEMM. Default on (A/B-verified): one barrier
    // per K8 window instead of two, with the next window's dequant/stores
    // overlapping the current window's tensor matmul.
    int fusedQ4DoubleBuf;
    // MNN_METAL_FUSED_Q4_SILU_MUL=0 keeps the gate/up group's MUL_SILU as its own
    // dispatch instead of folding it into the up projection's fused-Q4 M64
    // epilogue. Default on: the fold drops one full-tensor write plus one read.
    int fusedQ4SiluMul;
    // MNN_METAL_FUSED_Q4_GATEUP_DUAL=0 keeps gate and up as two prefill dispatches.
    // Default on: one dispatch accumulates both projections' N=32 tiles and applies
    // silu-mul in registers, so the gate tensor is never written or read back.
    // Once the fold drops the MUL_SILU pass the tile is compute-bound, and N=32
    // costs two narrower matmuls.
    int fusedQ4GateUpDual;
    // MNN_METAL_FUSED_Q4_GATEUP_DUAL_MIN_AREA: override the minimum prefill area
    // that enables the gate/up dual accumulator. Default 128: measured 2.9%
    // faster on Qwen3.5-2B s=128 than the separate-GEMM+MUL_SILU baseline.
    int fusedQ4GateUpDualMinArea;
    // MNN_METAL_H2D_QUEUED=0: restore the legacy drain+direct-write upload path.
    bool h2dQueued;
    // MNN_METAL_SPIN_WAIT=0: restore blocking waitUntilCompleted. Default on:
    // polling command-buffer status cuts the completion-wake latency at the cost
    // of one spinning CPU core while parked; falls back to blocking after ~20ms.
    // The core is burned for the whole wait, and a wait spans essentially a whole
    // token. Two attempts at bounding that burn were built and both falsified, so
    // the unconditional spin stands -- do not retry either without new tooling:
    //   1. Switch spinning off once the mean wait exceeds a ceiling (8000us).
    //      Forfeits the wake saving along with the burn; some shapes have long
    //      waits that still want spin, so wait length alone is not the signal.
    //   2. Sleep-poll the bulk of an EWMA-predicted wait, tight-spin only the
    //      last 1000us, keeping the wake saving at a fraction of the burn. It
    //      was unstable, not merely slower: macOS coalesces short nanosleeps, so
    //      a sleep overshoots the completion, inflating the measured wait,
    //      raising the EWMA, and lengthening the next sleep. Any retry needs a
    //      precise timer (mach_wait_until) and an estimator that its own
    //      overshoot cannot feed.
    bool spinWait;
    // MNN_METAL_COMMIT_NUM>0 overrides ops-per-commit cadence (device calibration).
    int commitNum;
    // MNN_METAL_DISABLE_REPLAY=1: disable encode replay (recorded command-list
    // re-emission for stable-shape forwards, see MetalReplay.hpp).
    bool replayDisabled;
    // MNN_METAL_REPLAY_DEBUG=1: log record/replay/invalidate transitions per op.
    bool replayDebug;

    // ---- diagnostics ----
    // MNN_METAL_OP_PROFILE_TIMELINE=<path>: dump per-op GPU timeline CSV
    // (requires -DMNN_METAL_OP_PROFILE=ON). nullptr when unset/empty.
    const char* opProfileTimeline;
    // MNN_METAL_OP_PROFILE_LEGACY=1: per-op command-buffer profile mode.
    bool opProfileLegacy;
    // MNN_METAL_PIPELINE_INFO=1: on every source-compiled pipeline, log the
    // kernel name, the driver's occupancy limits and the preprocessor macros it
    // was specialized with. The way to tell which shader variant a shape actually
    // routed to without guessing from the host-side gates.
    bool pipelineInfo;

    static const MetalEnv& get() {
        static const MetalEnv env = []{
            MetalEnv e;
            e.flashAttnPrefill = envTriState("MNN_ENABLE_FLASH_ATTN_PREFILL");
            e.prefillInshaderDequant = envTriState("MNN_METAL_PREFILL_INSHADER_DEQUANT_SGMATRIX");
            {
                const char* v = getenv("MNN_METAL_RESIZE_WAIT");
                e.resizeWaitMode = (v != nullptr && strcmp(v, "global") == 0) ? 1
                                 : (v != nullptr && strcmp(v, "none") == 0)   ? 2 : 0;
            }
            {
                const char* v = getenv("MNN_METAL_LN_TOKENS_PER_TG");
                e.lnRowParallel = (v == nullptr) ? 8 : atoi(v);
            }
            {
                const char* v = getenv("MNN_METAL_LN_SUM_REREAD");
                e.lnSumReread = (v == nullptr) ? 1 : atoi(v);
            }
            {
                const char* v = getenv("MNN_METAL_LN_TOKEN_LANE");
                e.lnTokenLane = (v == nullptr) ? 0 : atoi(v);
            }
            {
                const char* v = getenv("MNN_METAL_PREFILL_FA_SG");
                if (v == nullptr) {
                    e.prefillFaSg = -1; // unset: device-tiered auto threshold in MetalAttention
                } else {
                    e.prefillFaSg = (v[0] == '1') ? 1 : 0;
                }
            }
            e.fasgQReg = !envIs("MNN_METAL_FASG_Q_REG", '0');
            e.fasgNsg8 = !envIs("MNN_METAL_FASG_NSG8", '0');
            {
                const char* v = getenv("MNN_METAL_FASG_LOADBATCH");
                e.fasgLoadBatch = (v == nullptr) ? 4 : atoi(v);
            }
            e.faTcQRegDisabled = envIs("MNN_METAL_DISABLE_FATC_Q_REG", '1');
            e.faTcQkK32 = !envIs("MNN_METAL_FATC_QK_K32", '0');
            e.faTcOCt = !envIs("MNN_METAL_FATC_O_CT", '0');
            e.faTcKvDevTensor = !envIs("MNN_METAL_FATC_KV_DEV_TENSOR", '0');
            e.faTcQRev = !envIs("MNN_METAL_FATC_QREV", '0');
            e.faTcQCt = !envIs("MNN_METAL_FATC_Q_CT", '0');
            e.faTcPvK32 = !envIs("MNN_METAL_FATC_PV_K32", '0');
            {
                const char* v = getenv("MNN_METAL_FATC_DSPLIT");
                if (v == nullptr) {
                    e.faTcDSplit = -1; // unset: default resolved in MetalAttention
                } else {
                    int n = atoi(v);
                    e.faTcDSplit = n > 1 ? n : 1;
                }
            }
            {
                const char* v = getenv("MNN_METAL_ATTN_QSPLIT");
                if (v == nullptr) {
                    e.attnQSplit = 0; // unset: derive N from the scratch budget
                } else {
                    int n = atoi(v);
                    e.attnQSplit = n > 1 ? n : 1;
                }
            }
            {
                const char* v = getenv("MNN_METAL_ATTN_QSPLIT_MB");
                int n = (v == nullptr) ? 128 : atoi(v);
                e.attnQSplitMB = n > 1 ? n : 1;
            }
            {
                const char* v = getenv("MNN_METAL_PREFILL_FA_TENSORAPI");
                if (v == nullptr) {
                    e.prefillFaTensorApi = -1; // unset: default resolved in MetalAttention
                } else {
                    e.prefillFaTensorApi = (v[0] == '1') ? 1 : 0;
                }
            }
            {
                const char* v = getenv("MNN_METAL_DECODE_SDPA");
                if (v == nullptr) {
                    e.decodeSdpa = 1; // default auto-on (reuses split-KV threshold)
                } else {
                    int n = atoi(v);
                    e.decodeSdpa = n > 0 ? n : 0; // =0 explicit off
                }
            }
            {
                const char* v = getenv("MNN_METAL_DECODE_SDPA_NSG");
                if (v == nullptr) {
                    e.decodeSdpaNsg = 0; // 0 = device-tiered, resolved in MetalAttention
                } else {
                    int n = atoi(v);
                    e.decodeSdpaNsg = (n == 4 || n == 8 || n == 16 || n == 32) ? n : 0;
                }
            }
            {
                const char* v = getenv("MNN_METAL_DECODE_SDPA_QH_PER_TG");
                if (v == nullptr) {
                    e.decodeSdpaQhPerTg = 0; // 0 = auto, resolved in MetalAttention
                } else {
                    int n = atoi(v);
                    e.decodeSdpaQhPerTg = (n == 1 || n == 2 || n == 4 || n == 8) ? n : 0;
                }
            }
            {
                const char* v = getenv("MNN_METAL_DECODE_SDPA_NTG");
                if (v == nullptr) {
                    e.decodeSdpaNtg = 0; // 0 = auto, resolved in MetalAttention
                } else {
                    int n = atoi(v);
                    // powers of two only, so the interleaved kv stride stays regular
                    bool pow2 = n > 1 && (n & (n - 1)) == 0;
                    if (n == 1) {
                        e.decodeSdpaNtg = 1; // explicit single pass
                    } else {
                        e.decodeSdpaNtg = (pow2 && n <= 64) ? n : 0;
                    }
                }
            }
            e.lnFusionDisabled     = envIs("MNN_METAL_DISABLE_LN_FUSION", '1');
            e.gateUpFusionDisabled = envIs("MNN_METAL_DISABLE_GATE_UP_FUSION", '1');
            e.qkvFusionDisabled    = envIs("MNN_METAL_DISABLE_QKV_FUSION", '1');
            e.gemvW16Disabled = envIs("MNN_METAL_GEMV_W16", '0');
            e.lnStageFp32 = envIs("MNN_METAL_LN_STAGE_FP32", '1');
            e.lnStage = envTriState("MNN_METAL_LN_STAGE");
            e.qkvMerge = envTriState("MNN_METAL_QKV_MERGE");
            e.qkvMergedOut = envTriState("MNN_METAL_QKV_MERGED_OUT");
            e.qkvPackedGrid = envTriState("MNN_METAL_QKV_PACKED_GRID");
            e.ropeXCacheDisabled = envIs("MNN_METAL_DISABLE_ROPE_X_CACHE", '1');
            e.ropeTile = envTriState("MNN_METAL_ROPE_TILE");
            {
                const char* v = getenv("MNN_METAL_ROPE_TILE_T");
                e.ropeTileT = (v != nullptr) ? atoi(v) : 0;
            }
            e.gemvW16DsDisabled = envIs("MNN_METAL_GEMV_W16_DS", '0');
            e.gateUpLnStage = envIs("MNN_METAL_GATEUP_LN_STAGE", '1');
            {
                const char* v = getenv("MNN_METAL_GATEUP_SPLITK");
                int n = (v != nullptr) ? atoi(v) : -1;
                e.gateUpSplitK = (n == 0 || n == 1 || n == 2 || n == 4 || n == 8) ? n : -1;
            }
            {
                const char* v = getenv("MNN_METAL_GEMV_W16_MID");
                int n = (v != nullptr) ? atoi(v) : 0;
                e.gemvW16LanesPerBlock = (n == 1 || n == 2 || n == 4 || n == 8) ? n : 0;
            }
            {
                const char* v = getenv("MNN_METAL_QKV_SPLITK_MAX_BLOCKS");
                const int n = (v != nullptr) ? atoi(v) : 0;
                e.qkvSplitKMaxBlocks = n > 0 ? n : 16;
            }
            {
                const char* v = getenv("MNN_METAL_ENABLE_LMHEAD_SPLITK");
                const int n = (v != nullptr) ? atoi(v) : 0;
                e.lmheadSplitK = (v == nullptr) ? -1 : (n < 0 ? 0 : (n > 2 ? 2 : n));
            }
            e.w4w8OuterDequantGemm   = envIs("MNN_METAL_W4W8_OUTER_DEQUANT_GEMM_TENSORAPI", '1');
            e.fusedQ4Ksplit          = envTriState("MNN_METAL_FUSED_Q4_KSPLIT");
            e.fusedQ4M8              = envTriState("MNN_METAL_FUSED_Q4_M8");
            e.fusedQ4KsplitM8        = envTriState("MNN_METAL_FUSED_Q4_KSPLIT_M8");
            e.fusedQ4M64             = envTriState("MNN_METAL_FUSED_Q4_M64");
            e.fusedQ4SmemPad         = envTriState("MNN_METAL_FUSED_Q4_SMEM_PAD");
            e.fusedQ4Unorm           = envTriState("MNN_METAL_FUSED_Q4_UNORM");
            e.fusedQ4DoubleBuf       = envTriState("MNN_METAL_FUSED_Q4_DOUBLE_BUF");
            e.fusedQ4SiluMul         = envTriState("MNN_METAL_FUSED_Q4_SILU_MUL");
            e.fusedQ4GateUpDual      = envTriState("MNN_METAL_FUSED_Q4_GATEUP_DUAL");
            {
                const char* v = getenv("MNN_METAL_FUSED_Q4_GATEUP_DUAL_MIN_AREA");
                int n = (v != nullptr) ? atoi(v) : 128;
                e.fusedQ4GateUpDualMinArea = (n >= 64) ? n : 128;
            }
            e.h2dQueued            = !envIs("MNN_METAL_H2D_QUEUED", '0');
            e.spinWait             = !envIs("MNN_METAL_SPIN_WAIT", '0');
            {
                const char* v = getenv("MNN_METAL_COMMIT_NUM");
                e.commitNum = v != nullptr ? atoi(v) : 0;
            }
            e.replayDisabled = envIs("MNN_METAL_DISABLE_REPLAY", '1');
            e.replayDebug    = envIs("MNN_METAL_REPLAY_DEBUG", '1');
            {
                const char* v = getenv("MNN_METAL_OP_PROFILE_TIMELINE");
                e.opProfileTimeline = (v != nullptr && v[0] != '\0') ? v : nullptr;
            }
            e.opProfileLegacy = envIs("MNN_METAL_OP_PROFILE_LEGACY", '1');
            e.pipelineInfo    = envIs("MNN_METAL_PIPELINE_INFO", '1');
            return e;
        }();
        return env;
    }

private:
    static bool envIs(const char* name, char c) {
        const char* v = getenv(name);
        return v != nullptr && v[0] == c;
    }
    // "1" -> 1, "0" -> -1 (or 0 for flashAttnPrefill-style "explicit off"), unset -> 0/-1
    static int envTriState(const char* name) {
        const char* v = getenv(name);
        if (v == nullptr) return 0;
        return v[0] == '1' ? 1 : (v[0] == '0' ? -1 : 0);
    }
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalEnv_hpp */
