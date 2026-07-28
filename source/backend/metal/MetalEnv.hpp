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
    // MNN_METAL_QK_CAUSAL_TRI=0 disables causal-tri QK dispatch + causal-bound
    // softmax/AV. REQUIRED for non-causal masks (SWA / prefix LM / encoder).
    bool qkCausalTriOff;
    // MNN_ENABLE_FLASH_ATTN_PREFILL: -1 unset (config decides) / 0 force off / 1 force on.
    int flashAttnPrefill;
    // MNN_METAL_PREFILL_INSHADER_DEQUANT: 0 unset (size threshold decides) /
    // 1 force in-shader / -1 force outer-dequant.
    int prefillInshaderDequant;
    // MNN_METAL_RESIZE_WAIT: 0 "local" per-backend fence (default) /
    // 1 "global" legacy drain / 2 "none" skip both (experiment only).
    int resizeWaitMode;
    // MNN_METAL_DECODE_SPLITKV: kv-length threshold for split-KV decode attention.
    // Default 3072 (validated crossover on M5; M4 crossover measured ~1.2k (kv1024 flat, kv1493 +3.2%), kept
    // conservative pending iPhone/M3 calibration). =0 disables, =N>1 overrides.
    int decodeSplitKvThresh;

    // ---- fusion / misc ----
    // MNN_METAL_DISABLE_LN_FUSION set: disable LayerNorm+Conv1x1 fusion.
    bool lnFusionDisabled;
    // MNN_DISABLE_GATE_UP_FUSION set: disable Gate/Up leader/follower fusion.
    // (historical name without METAL prefix, kept for compatibility)
    bool gateUpFusionDisabled;
    // MNN_METAL_DISABLE_FUSED_Q4_GEMM=1: fall back to outer-dequant + fp GEMM.
    bool fusedQ4GemmDisabled;
    // MNN_METAL_GEMM_M64=1: experimental M=64 sg_matrix GEMM tile for the
    // outer-dequant prefill path on non-tensor-API devices (M4 calibration).
    bool gemmM64SgMatrix;
    // MNN_METAL_H2D_QUEUED=0: restore the legacy drain+direct-write upload path.
    bool h2dQueued;
    // MNN_METAL_COMMIT_NUM>0 overrides ops-per-commit cadence (device calibration).
    int commitNum;

    // ---- diagnostics ----
    // MNN_METAL_OP_PROFILE_TIMELINE=<path>: dump per-op GPU timeline CSV
    // (requires -DMNN_METAL_OP_PROFILE=ON). nullptr when unset/empty.
    const char* opProfileTimeline;
    // MNN_METAL_OP_PROFILE_LEGACY=1: per-op command-buffer profile mode.
    bool opProfileLegacy;

    static const MetalEnv& get() {
        static const MetalEnv env = []{
            MetalEnv e;
            e.qkCausalTriOff  = envIs("MNN_METAL_QK_CAUSAL_TRI", '0');
            e.flashAttnPrefill = envTriState("MNN_ENABLE_FLASH_ATTN_PREFILL");
            e.prefillInshaderDequant = envTriState("MNN_METAL_PREFILL_INSHADER_DEQUANT");
            {
                const char* v = getenv("MNN_METAL_RESIZE_WAIT");
                e.resizeWaitMode = (v != nullptr && strcmp(v, "global") == 0) ? 1
                                 : (v != nullptr && strcmp(v, "none") == 0)   ? 2 : 0;
            }
            {
                const char* v = getenv("MNN_METAL_DECODE_SPLITKV");
                if (v == nullptr) {
                    e.decodeSplitKvThresh = 3072;
                } else if (v[0] == '0') {
                    e.decodeSplitKvThresh = -1;
                } else {
                    int n = atoi(v);
                    e.decodeSplitKvThresh = n > 1 ? n : 3072;
                }
            }
            e.lnFusionDisabled     = getenv("MNN_METAL_DISABLE_LN_FUSION") != nullptr;
            e.gateUpFusionDisabled = getenv("MNN_DISABLE_GATE_UP_FUSION") != nullptr;
            e.fusedQ4GemmDisabled  = envIs("MNN_METAL_DISABLE_FUSED_Q4_GEMM", '1');
            e.gemmM64SgMatrix      = envIs("MNN_METAL_GEMM_M64", '1');
            e.h2dQueued            = !envIs("MNN_METAL_H2D_QUEUED", '0');
            {
                const char* v = getenv("MNN_METAL_COMMIT_NUM");
                e.commitNum = v != nullptr ? atoi(v) : 0;
            }
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
