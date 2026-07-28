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

ErrorCode MetalExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
#ifdef MNN_SESSION_CPU_TRACE
    auto t0 = std::chrono::steady_clock::now();
#endif
    auto encoder           = backend->encoder_for_net();
#if MNN_METAL_OP_PROFILE
    backend->profileMarkOp(this);
#endif
    this->onEncode(inputs, outputs, encoder);
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
