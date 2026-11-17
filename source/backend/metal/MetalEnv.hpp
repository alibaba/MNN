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

// Single registry for every MNN Metal env switch; parsed once on first use.
// To add a switch, extend this struct (no scattered getenv), document it in
// skills/metal-optimize/env-registry.md, and prefer DISABLE_/ENABLE_ naming.
struct MetalEnv {
    // ---- performance path ----
    // MNN_METAL_RESIZE_WAIT: 0 "local" per-backend fence (default) /
    // 1 "global" legacy drain / 2 "none" skip both (experiment only).
    int resizeWaitMode;
    // MNN_METAL_FASG_LOADBATCH=N: V-fragment batch width (loads issued before
    // any MMA consumes one); 0 disables. Loops must stay fully unrolled.
    int fasgLoadBatch;
    // MNN_METAL_ATTN_QSPLIT_MB: per-piece budget (MB) for the three-stage
    // prefill scratch; the q sequence is split to fit it. Default 128.
    int attnQSplitMB;
    // MNN_METAL_LN_TOKENS_PER_TG: tokens per threadgroup in the C4 RMSNorm at
    // prefill. Default 8, auto-on at outside>=256.
    int lnRowParallel;
    // ---- fusion / misc ----
    // MNN_METAL_DECODE_SDPA_KVB: kv tokens per stream-loop iteration in the
    // decode SDPA kernel. 0=auto, 1=serial, 2/4=overlap form; 4 clamps to 2
    // at head_dim > 128 (register pressure).
    int decodeSdpaKvb;
    // MNN_METAL_QKV_SPLITK_MAX_BLOCKS: quant-block ceiling for the plain
    // split; block count stands in for how parallelism-starved the dispatch is.
    int qkvSplitKMaxBlocks;
    // MNN_METAL_FUSED_Q4_GATEUP_DUAL_MIN_AREA: minimum prefill area that
    // enables the gate/up dual accumulator. Default 128.
    int fusedQ4GateUpDualMinArea;
    // MNN_METAL_H2D_QUEUED=0: restore the legacy drain+direct-write upload path.
    bool h2dQueued;
    // MNN_METAL_SPIN_WAIT=0: use blocking waitUntilCompleted. Default polls,
    // falling back to blocking after ~20ms.
    bool spinWait;
    // MNN_METAL_QOS_BOOST=0: disable the user-interactive QoS override around
    // encode/commit so the scheduler cannot park the caller on efficiency cores.
    bool qosBoost;
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
    // MNN_METAL_PIPELINE_INFO=1: log kernel name, occupancy limits, and
    // specialization macros for every source-compiled pipeline.
    bool pipelineInfo;

    static const MetalEnv& get() {
        static const MetalEnv env = []{
            MetalEnv e;
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
                const char* v = getenv("MNN_METAL_FASG_LOADBATCH");
                e.fasgLoadBatch = (v == nullptr) ? 4 : atoi(v);
            }
            {
                const char* v = getenv("MNN_METAL_ATTN_QSPLIT_MB");
                int n = (v == nullptr) ? 128 : atoi(v);
                e.attnQSplitMB = n > 1 ? n : 1;
            }
            {
                const char* v = getenv("MNN_METAL_QKV_SPLITK_MAX_BLOCKS");
                const int n = (v != nullptr) ? atoi(v) : 0;
                e.qkvSplitKMaxBlocks = n > 0 ? n : 16;
            }
            {
                const char* v = getenv("MNN_METAL_DECODE_SDPA_KVB");
                const int n = (v != nullptr) ? atoi(v) : 0;
                e.decodeSdpaKvb = (n == 1 || n == 2 || n == 4) ? n : 0;
            }
            {
                const char* v = getenv("MNN_METAL_FUSED_Q4_GATEUP_DUAL_MIN_AREA");
                int n = (v != nullptr) ? atoi(v) : 128;
                e.fusedQ4GateUpDualMinArea = (n >= 64) ? n : 128;
            }
            e.h2dQueued            = !envIs("MNN_METAL_H2D_QUEUED", '0');
            e.spinWait             = !envIs("MNN_METAL_SPIN_WAIT", '0');
            e.qosBoost             = !envIs("MNN_METAL_QOS_BOOST", '0');
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
    // "1" -> 1, "0" -> -1, unset -> 0/-1
    static int envTriState(const char* name) {
        const char* v = getenv(name);
        if (v == nullptr) return 0;
        return v[0] == '1' ? 1 : (v[0] == '0' ? -1 : 0);
    }
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalEnv_hpp */
