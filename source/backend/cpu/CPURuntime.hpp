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
    std::vector<CPUGroup> groups;
 // RVV attributes
    bool rvv = false;
    int rvv_vlen = 0;
    int rvv_version = 0;
    bool zvfh = false;
    bool zvkn = false;

    int cpuNumber = 0;
    int smeCoreNumber = 0;
};

using cpu_mask_t = unsigned long;
int MNNSetSchedAffinity(const int* cpuIDs, int size);
int MNNGetCurrentPid();
cpu_mask_t MNNGetCPUMask(const std::vector<int>& cpuIds);
const MNNCPUInfo* MNNGetCPUInfo();

#endif /* CPUInfo_hpp */
