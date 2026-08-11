//
//  MetalExecution.mm
//  MNN
//
//  Created by MNN on 2023/11/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MetalExecution.hpp"
#import "backend/metal/MetalBackend.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#if MNN_METAL_ENABLED
namespace MNN {

#ifdef MNN_SESSION_CPU_TRACE
MetalCpuTrace::~MetalCpuTrace() {
    if (encodeOps.load() > 0 || waitCalls.load() > 0 || commitCalls.load() > 0) {
        printf("\n===== Metal CPU Trace =====\n");
        printf("encode: %.3f ms total, %llu ops, %.3f us/op\n",
               encodeNs.load() / 1e6, (unsigned long long)encodeOps.load(),
               encodeOps.load() ? encodeNs.load() / 1e3 / encodeOps.load() : 0.0);
        printf("commit: %.3f ms total, %llu calls, %.3f us/call (mid-exec flush)\n",
               commitNs.load() / 1e6, (unsigned long long)commitCalls.load(),
               commitCalls.load() ? commitNs.load() / 1e3 / commitCalls.load() : 0.0);
        printf("wait  : %.3f ms total, %llu calls, %.3f us/call\n",
               waitNs.load() / 1e6, (unsigned long long)waitCalls.load(),
               waitCalls.load() ? waitNs.load() / 1e3 / waitCalls.load() : 0.0);
        static const char* kSiteNames[4] = {"resizeFence", "copyD2H", "copyH2D", "onSync"};
        for (int i = 0; i < 4; ++i) {
            if (waitSiteCalls[i].load() > 0) {
                printf("  wait[%s]: %.3f ms, %llu calls, %.3f us/call\n", kSiteNames[i],
                       waitSiteNs[i].load() / 1e6, (unsigned long long)waitSiteCalls[i].load(),
                       waitSiteNs[i].load() / 1e3 / waitSiteCalls[i].load());
            }
        }
        if (gpuBuffers.load() > 0) {
            printf("gpu   : busy %.3f ms, gap %.3f ms, %llu cmd buffers, busy %.3f us/buf\n",
                   gpuBusyNs.load() / 1e6, gpuGapNs.load() / 1e6,
                   (unsigned long long)gpuBuffers.load(),
                   gpuBusyNs.load() / 1e3 / gpuBuffers.load());
        }
        printf("===========================\n");
    }
}
MetalCpuTrace& metalCpuTrace() {
    static MetalCpuTrace trace;
    return trace;
}
#endif // MNN_SESSION_CPU_TRACE

MetalExecution::MetalExecution(Backend *backend) : Execution(backend) {
    // Do nothing
}

// FNV-1a over the current device buffer+offset of every input/output tensor:
// a cheap fingerprint of "this encode would bind the same addresses as last
// time". Two consecutive identical fingerprints arm the recording pass.
static uint64_t _replayHashIO(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    for (auto t : inputs) {
        auto b = MetalBackend::getBuffer(t);
        mix((uint64_t)(__bridge void*)b.first);
        mix((uint64_t)b.second);
    }
    for (auto t : outputs) {
        auto b = MetalBackend::getBuffer(t);
        mix((uint64_t)(__bridge void*)b.first);
        mix((uint64_t)b.second);
    }
    return h;
}

ErrorCode MetalExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
#ifdef MNN_SESSION_CPU_TRACE
    auto t0 = std::chrono::steady_clock::now();
#endif
    auto encoder           = backend->encoder_for_net();
#if MNN_METAL_OP_PROFILE
    backend->profileMarkOp(this);
#endif
    // Encode-replay path (see MetalReplay.hpp). Disabled in OP_PROFILE builds:
    // profileNextSubpass swaps the encoder mid-op, which recording can't model.
#if !MNN_METAL_OP_PROFILE
    const bool replayAllowed = !MetalEnv::get().replayDisabled && this->canRecordEncode();
#else
    const bool replayAllowed = false;
#endif
    bool encoded = false;
    if (replayAllowed) {
        if (mReplayState == 1) {
            if (this->onReplayUpdate(inputs, outputs) && metalReplayEmit(mReplayEvents, encoder)) {
                encoded = true;
                mReplayFailCount = 0;
            } else {
                // Stale recording (address moved or dynamic hook bailed):
                // drop it and re-observe; the normal encode below re-arms.
                mReplayState  = 0;
                mReplayStable = 0;
                mReplayEvents.clear();
                // Normal invalidations (KV expansion, one-off prefill between
                // decode phases) recover on the next record; a recording that
                // keeps failing back-to-back is poison — ban it.
                if (++mReplayFailCount >= 8) {
                    mReplayState = -1;
                    if (MetalEnv::get().replayDebug) {
                        MNN_PRINT("[MetalReplay] ban exe=%p (repeated replay failure)\n", this);
                    }
                }
                if (MetalEnv::get().replayDebug) {
                    MNN_PRINT("[MetalReplay] invalidate exe=%p, fallback to normal encode\n", this);
                }
            }
        }
        if (!encoded && mReplayState == 0) {
            uint64_t key = _replayHashIO(inputs, outputs);
            if (key == mReplayKey) {
                mReplayStable++;
            } else {
                mReplayKey    = key;
                mReplayStable = 0;
            }
            if (mReplayStable >= 2) {
                MetalReplayProxy* proxy = [[MetalReplayProxy alloc] initWithTarget:encoder events:&mReplayEvents];
                gMetalReplayProxy = proxy;
                this->onEncode(inputs, outputs, (id<MTLComputeCommandEncoder>)proxy);
                gMetalReplayProxy = nil;
                if (proxy.failed) {
                    mReplayState = -1; // unrecordable encode: stay on normal path
                    mReplayEvents.clear();
                    if (MetalEnv::get().replayDebug) {
                        MNN_PRINT("[MetalReplay] ban exe=%p (unsupported encoder call)\n", this);
                    }
                } else {
                    mReplayState = 1;
                    if (MetalEnv::get().replayDebug) {
                        MNN_PRINT("[MetalReplay] record exe=%p events=%zu\n", this, mReplayEvents.size());
                    }
                }
                encoded = true;
            }
        }
    }
    if (!encoded) {
        this->onEncode(inputs, outputs, encoder);
    }
#if MNN_METAL_OP_PROFILE
    // counter mode: per-op encoder boundary (cheap, same command buffer)
    backend->profileOpEncoded();
#endif
#ifdef MNN_SESSION_CPU_TRACE
    auto t1 = std::chrono::steady_clock::now();
    metalCpuTrace().encodeNs += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    metalCpuTrace().encodeOps += 1;
#endif
    if(backend->isCmdBufferCommit()) {
#ifdef MNN_SESSION_CPU_TRACE
        auto t2 = std::chrono::steady_clock::now();
#endif
        backend->flushEncoder();
        backend->commit_net();
#ifdef MNN_SESSION_CPU_TRACE
        auto t3 = std::chrono::steady_clock::now();
        metalCpuTrace().commitNs += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        metalCpuTrace().commitCalls += 1;
#endif
    }

    return NO_ERROR;
}


};
#endif
