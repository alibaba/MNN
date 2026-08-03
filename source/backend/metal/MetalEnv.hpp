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
    // tensor API (prefill_flash_attn_nax). Keeps S and O in registers for the
    // whole KV sweep so the O(n^2) score matrix never reaches global memory,
    // unlike the three-stage path.
    //  -1 (unset) = default resolved from the data-driven mCausalLayout flag
    //       (standard causal mask => on, arbitrary mask => off). Arbitrary masks
    //       can never enable it since this kernel hard-codes causal.
    //   1 = explicit on (still gated to causal layout in MetalAttention)
    //   0 = explicit off, legacy prefill paths untouched
    // Further gated by isSupportTensorCoopInput() (M5+ only), so on
    // M4/M3/iPhone the default is a no-op. M5 rep5: pp512 +4.2% / pp1024
    // +5.5% / pp2048 +9.0% / pp4096 +17%.
    int prefillFaTensorApi;
    // MNN_METAL_DECODE_SDPA: single-pass fused decode attention (roadmap #20
    // restart). decode_splitkv with nwg pinned to 1, no reduce dispatch, final
    // output written by the kernel itself (MLX sdpa_vector form). It replaces
    // the split-KV path on the same eligibility/threshold, so it defaults ON
    // (auto) just like split-KV, which is already default-on for all devices.
    // unset = 1 (auto kv threshold, same clamped threshold as split-KV);
    // 0 = explicit off (legacy split-KV path); N>1 = explicit kv threshold
    // override (probe). M4 Pro paired rep5: 0.6B p2048 +6.0% / p4096 +7.5%
    // (nsg32); M5 p2048 +6.2% (nsg8). See decodeSdpaNsg for the device tier.
    int decodeSdpa;
    // MNN_METAL_DECODE_SDPA_NSG: simdgroups per threadgroup for the single-pass
    // kernel. Allowed {4, 8, 16, 32}. Default 0 = device-tiered (resolved in
    // MetalAttention): tensor-API/M5 -> 8 (M5 sweep p2048 nsg8 +6.2% vs nsg32
    // +3.3%); non-tensor-API/M4-class -> 32 (M4 Pro paired p2048 nsg32 +6.0%
    // vs nsg8 -3.5%, opposite of M5). M1/M2/M3/iPhone uncalibrated, inherit the
    // M4 non-tensor branch (nsg32) pending on-device sweep.
    int decodeSdpaNsg;


    // ---- fusion / misc ----
    // MNN_METAL_DISABLE_LN_FUSION=1: disable LayerNorm+Conv1x1 fusion.
    bool lnFusionDisabled;
    // MNN_METAL_DISABLE_GATE_UP_FUSION=1: disable Gate/Up leader/follower fusion.
    bool gateUpFusionDisabled;
    // MNN_METAL_DISABLE_QKV_FUSION=1: disable Q/K/V leader/follower fusion.
    bool qkvFusionDisabled;
    // MNN_METAL_GEMV_SPLITK: decode GEMV K-split (SPLIT_K_2 variant of the 2sg
    // kernel: 4 simdgroups per tg, two K-halves per row + tg reduce).
    // 0 = off (legacy 2sg, 64 threads), unset/1 = on (default).
    int gemvSplitK;
    // MNN_METAL_LINEAR_ATTN_SGMM: simdgroup_matrix (8x8 MMA) chunked prefill
    // kernel for LinearAttention on non-tensor-API devices (M4-class/iPhone).
    // Replaces the per-timestep scalar fused_chunk_sg path for seq >= 16.
    // 0 = off (legacy scalar chunk path), unset/1 = on (default).
    int linearAttnSgmm;
    // MNN_METAL_W4W8_OUTER_DEQUANT_GEMM_TENSORAPI=1: take the outer-dequant +
    // fp GEMM path instead of the fused Q4/Q8 GEMM that unpacks weights
    // in-kernel (A/B baseline + emergency rollback). Only meaningful on
    // tensor-API devices (M5+), where the fused path is the default.
    bool w4w8OuterDequantGemm;
    // MNN_METAL_H2D_QUEUED=0: restore the legacy drain+direct-write upload path.
    bool h2dQueued;
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

            e.lnFusionDisabled     = envIs("MNN_METAL_DISABLE_LN_FUSION", '1');
            e.gateUpFusionDisabled = envIs("MNN_METAL_DISABLE_GATE_UP_FUSION", '1');
            e.qkvFusionDisabled    = envIs("MNN_METAL_DISABLE_QKV_FUSION", '1');
            {
                const char* v = getenv("MNN_METAL_GEMV_SPLITK");
                e.gemvSplitK = 1;
                if (v != nullptr) {
                    int n = atoi(v);
                    e.gemvSplitK = (n < 0) ? 0 : (n > 2 ? 2 : n);
                }
            }
            {
                const char* v = getenv("MNN_METAL_LINEAR_ATTN_SGMM");
                e.linearAttnSgmm = (v != nullptr && v[0] == '0') ? 0 : 1;
            }
            e.w4w8OuterDequantGemm   = envIs("MNN_METAL_W4W8_OUTER_DEQUANT_GEMM_TENSORAPI", '1');
            e.h2dQueued            = !envIs("MNN_METAL_H2D_QUEUED", '0');
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
