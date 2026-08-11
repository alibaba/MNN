//
//  CPURuntime.hpp
//  MNN
//
//  Created by MNN on 2018/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef CPURuntime_hpp
#define CPURuntime_hpp

#include <stdint.h>
#include <vector>
#include "core/Macro.h"
struct CPUGroup {
    uint32_t minFreq;
    uint32_t maxFreq;
    std::vector<int> ids;
};
struct MNNCPUInfo {
    bool fp16arith = false;
    bool dot = false;
    bool i8mm = false;
    bool sve2 = false;
    bool sme2 = false;
#if defined(MNN_SME2) && defined(MNN_SUPPORT_TRANSFORMER_FUSE)
    bool fp16fml = false;
#endif
    std::vector<CPUGroup> groups;
    // RISC-V Vector features, as reported by Linux riscv_hwprobe(2). The probe
    // asks for no specific CPU set, so the kernel ANDs every bit across all
    // online CPUs: runtime dispatch stays valid when a thread migrates.
    bool rvv = false;
    bool zvfh = false;
    bool zvfhmin = false;

    int cpuNumber = 0;
    int smeCoreNumber = 0;
};

using cpu_mask_t = unsigned long;
int MNNSetSchedAffinity(const int* cpuIDs, int size);
int MNNGetCurrentPid();
cpu_mask_t MNNGetCPUMask(const std::vector<int>& cpuIds);
const MNNCPUInfo* MNNGetCPUInfo();

#endif /* CPUInfo_hpp */
