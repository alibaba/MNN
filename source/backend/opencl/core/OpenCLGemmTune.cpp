//
//  OpenCLGemmTune.cpp
//  MNN
//
//  Created by MNN on 2024/05/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include <algorithm>
#include <string>
#include <math.h>
#include <vector>
#include "core/Macro.h"

namespace MNN {
namespace OpenCL {

    
static void generateCombinations(const std::vector<std::vector<uint32_t>> &candidates, std::vector<uint32_t> &currentCombination, std::vector<std::vector<uint32_t>> &totalCombinations, int depth) {
    if (depth == candidates.size()) {
        totalCombinations.emplace_back(currentCombination);
        return;
    }

    // Loop all candidates
    for (int i = 0; i < candidates[depth].size(); i++) {
        currentCombination[depth] = candidates[depth][i];
        // Recurrence
        generateCombinations(candidates, currentCombination, totalCombinations, depth + 1);
    }
}
    
    
static bool isCandidateValid(uint32_t kwg, uint32_t kwi, uint32_t mwg, uint32_t mdimc, uint32_t vwm, uint32_t nwg, uint32_t ndimc, uint32_t vwn, uint32_t mdima, uint32_t ndimb, uint32_t sa, uint32_t sb, OpenCLRuntime *runtime, const std::vector<uint32_t>& gemmSize) {
    // problem size align
    if(gemmSize[0] % mwg != 0 || gemmSize[1] % nwg != 0) {
        return false;
    }

    if(mwg % (mdimc * vwm) != 0 || mwg % (mdima * vwm) != 0) {
        return false;
    }
    if(nwg % (ndimc * vwn) != 0 || nwg % (ndimb * vwn) != 0) {
        return false;
    }
    uint32_t kdima = (mdimc * ndimc) / mdima;
    uint32_t kdimb = (mdimc * ndimc) / ndimb;
    if(sa == 1 || sb == 1) {
        // params align
        if(kwg % kwi != 0) {
            return false;
        }
        if(kwg % kdima != 0 || kwg % kdimb != 0) {
            return false;
        }
        if(gemmSize[2] % kwg != 0) {
            return false;
        }
    }

    if(mdimc != mdima || ndimc != ndimb) {
        return false;
    }
    if(sa != sb) {
        return false;
    }
    
    // no local memory no need tune kwg
    if(sa == 0 && sb == 0 && kwg == 32) {
        return false;
    }
    
    // local memory limit
    uint32_t local_mem_size = 0;
    if(sa) {
        local_mem_size += kwg * mwg;
    }
    if(sb) {
        local_mem_size += kwg * nwg;
    }
    if(runtime->isSupportedFP16()) {
        local_mem_size *= 2;
    } else {
        local_mem_size *= 4;
    }
    if(local_mem_size > runtime->getMaxLocalMem()) {
        return false;
    }
    
    // local size limit
    if(mdimc * ndimc > runtime->MaxWorkGroupSize()) {
        return false;
    }
    
    // reduce total candidate number
    if(mdimc != mdima || ndimc != ndimb) {
        return false;
    }
    
    
    bool totalLarge = 1.0 * gemmSize[0] / 1024 * gemmSize[1] / 1024 * gemmSize[2] / 1024 >= 0.5;
    bool dimLarge = gemmSize[0] > 128 && gemmSize[1] > 128 && gemmSize[2] > 128;
    if(gemmSize[4] == 1) {
        if(totalLarge && dimLarge) {
            if(mwg * nwg < 128 * 64) {
                return false;
            }
            if(mdimc * ndimc < 16 * 8) {
                return false;
            }
            if(vwm * vwn < 4 * 4) {
                return false;
            }
        } else {
            if(mwg * nwg > 128 * 64) {
                return false;
            }
            if(mdimc * ndimc > 16 * 8) {
                return false;
            }
            if(vwm * vwn > 4 * 4) {
                return false;
            }
        }
    }
    
    return true;
}
    
std::vector<uint32_t> getGemmParams(const std::vector<uint32_t> &gemmSize, const std::vector<cl::Buffer> tensorMemory,
                                       OpenCLRuntime *runtime) {
    MNN_ASSERT(gemmSize.size() == 6); // M, N, K, Layout, Batch, Bias
    MNN_ASSERT(gemmSize[0] % 16 == 0);
    MNN_ASSERT(gemmSize[1] % 16 == 0);
    MNN_ASSERT(gemmSize[2] % 4 == 0);

    MNN_ASSERT((gemmSize[5] == 0 && tensorMemory.size() == 3) || (gemmSize[5] == 1 && tensorMemory.size() == 4));
    auto& tunedGemmParams = runtime->tunedGemmParamsMap();
    
    std::vector<uint32_t> info(gemmSize);
    uint32_t isFp16 = runtime->isSupportedFP16();
    info.emplace_back(isFp16);
    
    if (tunedGemmParams.find(info) != tunedGemmParams.end()) {
        return tunedGemmParams[info];
    }
    
    if(runtime->getCLTuneLevel() == None) {
        auto getMaxDivisor = [](uint32_t num) -> uint32_t {
            std::vector<int> divisors = {128, 64, 32};
            for (const auto& divisor : divisors) {
                if (num % divisor == 0) {
                    return divisor;
                }
            }
            return 16;
        };
        float multiNum = 1.0 * gemmSize[0] / 512.0 * gemmSize[1] / 512.0 * gemmSize[2] / 512.0;
        int maxDivsorM = getMaxDivisor(gemmSize[0]);
        int maxDivsorN = getMaxDivisor(gemmSize[1]);
        
        if(gemmSize[4] == 1) {// Gemm
            if(gemmSize[0] >= 256 && gemmSize[1] >= 256 && gemmSize[2] >= 256) {
                if(multiNum > 8.0) {
                    if(maxDivsorM >= 128 && maxDivsorN >= 64) {
                        return {16, 2, 16, 16, 128, 8, 8, 64, 0, 0, 0, 1, 8, 8};
                    }
                }
                if(maxDivsorM >= 64 && maxDivsorN >= 64) {
                    return {16, 2, 8, 8, 64, 8, 8, 64, 0, 0, 0, 1, 8, 8};
                }
            }
        } else {// BatchGemm
            if(maxDivsorM >= 64 && maxDivsorN >= 128) {
                return {16, 2, 16, 16, 64, 8, 8, 128, 0, 0, 1, 0, 2, 8};
            } else if(maxDivsorM >= 64 && maxDivsorN >= 64) {
                return {16, 2, 8, 8, 64, 8, 8, 64, 0, 0, 1, 0, 4, 4};
            }
        }
        return {16, 2, 4, 4, 16, 4, 4, 16, 0, 0, 1, 0, 2, 2};
    }

    std::vector<std::vector<uint32_t>> totalCombinations; // save total candidate combinations
    std::vector<uint32_t> params_prefer = {16, 2, 4, 4, 16, 4, 4, 16, 0, 0, 1, 0, 2, 2};
    totalCombinations.emplace_back(params_prefer);
    uint32_t min_cost = UINT_MAX;
    
    if(runtime->getCLTuneLevel() >= Wide) {
        // set candidates=
        totalCombinations.push_back({16, 2, 16, 16, 64 , 8 , 8 , 128, 0, 0, 0, 0, 4, 8});//12
        totalCombinations.push_back({16, 2, 16, 16, 128, 8 , 8 , 64 , 0, 0, 0, 0, 8, 8});//11 ..
        totalCombinations.push_back({16, 2, 16, 16, 128, 16, 16, 128, 0, 0, 0, 0, 8, 8});//1
        totalCombinations.push_back({16, 2, 16, 16, 128, 8 , 8 , 32 , 0, 0, 0, 1, 8, 4});//1
        totalCombinations.push_back({16, 2, 8 , 8 , 16 , 8 , 8 , 64, 0, 0, 0, 0, 2, 8});
        totalCombinations.push_back({16, 2, 16, 16, 64 , 8 , 8 , 128, 0, 0, 0, 1, 4, 8});//10

        totalCombinations.push_back({16, 2, 16, 16, 64 , 8 , 8 , 32 , 0, 0, 0, 0, 4, 4});
        totalCombinations.push_back({16, 2, 8,  8 , 32 , 8 , 8 , 128, 0, 0, 1, 0, 2, 8});//2
        totalCombinations.push_back({16, 2, 16, 16, 64 , 8 , 8 , 128, 0, 0, 1, 1, 2, 8});//12
        totalCombinations.push_back({16, 2, 16, 16, 128, 8 , 8 , 64 , 0, 0, 1, 1, 2, 8});//2
        totalCombinations.push_back({16, 2, 16, 16, 128, 8 , 8 , 128, 0, 0, 0, 0, 8, 8});
        totalCombinations.push_back({16, 2, 8 , 8 , 16 , 8 , 8 , 128, 0, 0, 0, 0, 2, 8});

        if(runtime->getCLTuneLevel() < Fast) {
            totalCombinations.push_back({16, 2, 16, 16, 128, 8 , 8 , 64 , 0, 0, 1, 0, 8, 8});//4
            totalCombinations.push_back({16, 2, 16, 16, 128, 8 , 8 , 64 , 0, 0, 0, 1, 8, 8});//6
            totalCombinations.push_back({16, 2, 16, 16, 128, 8 , 8 , 64 , 0, 0, 1, 1, 8, 8});//4
    
            totalCombinations.push_back({16, 2, 16, 16, 128, 8 , 8 , 64 , 0, 0, 1, 0, 2, 8});//3
            totalCombinations.push_back({16, 2, 8,  8 , 64 , 8 , 8 , 64 , 0, 0, 1, 0, 2, 8});//1
            totalCombinations.push_back({16, 2, 16, 16, 128, 8 , 8 , 64 , 0, 0, 1, 1, 4, 4});//1
            totalCombinations.push_back({16, 2, 16, 16, 64 , 8 , 8 , 128, 0, 0, 1, 0, 2, 8});//3
            
            totalCombinations.push_back({16, 2, 16, 16, 128, 8 , 8 , 32 , 0, 0, 0, 0, 4, 4});//1
            totalCombinations.push_back({16, 2, 16, 16, 128, 16, 16, 128, 0, 0, 0, 1, 8, 8});//2
            totalCombinations.push_back({16, 2, 16, 16, 128, 16, 16, 128, 0, 0, 1, 0, 8, 8});//1
            totalCombinations.push_back({16, 2, 8 , 8 , 16 , 8 , 8 , 128, 0, 0, 1, 0, 2, 8});//1
            totalCombinations.push_back({16, 2, 8 , 8 , 16 , 8 , 8 , 128, 0, 0, 1, 1, 2, 8});//1
            
            totalCombinations.push_back({16, 2, 16, 16, 64 , 8 , 8 , 32 , 0, 0, 0, 1, 4, 4});//1
            totalCombinations.push_back({16, 2, 16, 16, 64 , 8 , 8 , 32 , 0, 0, 1, 0, 4, 4});
            totalCombinations.push_back({16, 2, 16, 16, 128, 8 , 8 , 64 , 0, 0, 1, 0, 4, 8});
            totalCombinations.push_back({16, 2, 16, 16, 128, 8 , 8 , 128, 0, 0, 0, 1, 8, 8});
            totalCombinations.push_back({16, 2, 16, 16, 128, 8 , 8 , 128, 0, 0, 1, 1, 8, 8});
            
            totalCombinations.push_back({16, 2, 8, 8, 32, 8, 8, 32, 0, 0, 1, 0, 2, 4});
            totalCombinations.push_back({16, 2, 8, 8, 16, 8, 8, 32, 0, 0, 1, 1, 2, 4});
        }
    } else {
        // get all combinations
        std::vector<std::vector<uint32_t>> candidates = {
            {16, 32},         // KWG
            {2},              // KWI
            {8, 16},          // MDIMA
            {8, 16},          // MDIMC
            {16, 32, 64, 128}, // MWG
            {8, 16},          // NDIMB
            {8, 16},          // NDIMC
            {16, 32, 64, 128}, // NWG
            {0},              // SA
            {0},              // SB
            {0, 1},           // STRM
            {0, 1},           // STRN
            {2, 4, 8},        // VWM
            {2, 4, 8}        // VWN
        };
        
        std::vector<uint32_t> currentCombination(candidates.size());
        generateCombinations(candidates, currentCombination, totalCombinations, 0);
    }
    for(int i = 0; i < totalCombinations.size(); i++) {
        uint32_t kwg   = totalCombinations[i][0];
        uint32_t kwi   = totalCombinations[i][1];
        uint32_t mdima = totalCombinations[i][2];
        uint32_t mdimc = totalCombinations[i][3];
        uint32_t mwg   = totalCombinations[i][4];
        uint32_t ndimb = totalCombinations[i][5];
        uint32_t ndimc = totalCombinations[i][6];
        uint32_t nwg   = totalCombinations[i][7];
        uint32_t sa    = totalCombinations[i][8];
        uint32_t sb    = totalCombinations[i][9];
        uint32_t strm  = totalCombinations[i][10];
        uint32_t strn  = totalCombinations[i][11];
        uint32_t vwm   = totalCombinations[i][12];
        uint32_t vwn   = totalCombinations[i][13];
        
        if(isCandidateValid(kwg, kwi, mwg, mdimc, vwm, nwg, ndimc, vwn, mdima, ndimb, sa, sb, runtime, gemmSize)) {
            
            std::set<std::string> buildOptions;
            buildOptions.clear();
            buildOptions.emplace("-DKWG="   + std::to_string(kwg));
            buildOptions.emplace("-DKWI="   + std::to_string(kwi));
            buildOptions.emplace("-DMDIMA=" + std::to_string(mdima));
            buildOptions.emplace("-DMDIMC=" + std::to_string(mdimc));
            buildOptions.emplace("-DMWG="   + std::to_string(mwg));
            buildOptions.emplace("-DNDIMB=" + std::to_string(ndimb));
            buildOptions.emplace("-DNDIMC=" + std::to_string(ndimc));
            buildOptions.emplace("-DNWG="   + std::to_string(nwg));
            buildOptions.emplace("-DSA="    + std::to_string(sa));
            buildOptions.emplace("-DSB="    + std::to_string(sb));
            buildOptions.emplace("-DSTRM="  + std::to_string(strm));
            buildOptions.emplace("-DSTRN="  + std::to_string(strn));
            buildOptions.emplace("-DVWM="   + std::to_string(vwm));
            buildOptions.emplace("-DVWN="   + std::to_string(vwn));
            
            if(gemmSize[3] >= 4) {
                buildOptions.emplace(" -DOUTPUTMN");
            }
            if(runtime->getGpuType() == GpuType::ADRENO) {
                buildOptions.emplace(" -DUSE_CL_MAD=1");
                buildOptions.emplace(" -DRELAX_WORKGROUP_SIZE=1");
            }
            
            if(gemmSize[5] == 1) {
                buildOptions.emplace(" -DBIAS");
            }

            int localM = mdimc;
            int localN = ndimc;
            
            std::shared_ptr<KernelWrap> kernel = runtime->buildKernel("matmul_params_buf", "Xgemm", buildOptions);
            if(kernel == nullptr) {
                continue;
            }
            if(gemmSize[4] > 1) {
                kernel =    runtime->buildKernel("matmul_params_buf", "XgemmBatched", buildOptions);
                if(kernel == nullptr) {
                    continue;
                }
            }
            
            if(localM * localN > runtime->getMaxWorkGroupSize(kernel)) {
                continue;
            }
            int tileM = mwg;
            int tileN = nwg;
            int out_per_thread_m = tileM / localM;
            int out_per_thread_n = tileN / localN;
            
            std::vector<uint32_t>  globalWorkSize = {static_cast<uint32_t>(gemmSize[0]/out_per_thread_m), static_cast<uint32_t>(gemmSize[1]/out_per_thread_n), gemmSize[4]};
            std::vector<uint32_t>  localWorkSize = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN), 1};
            
            float alpha = 1.0;
            float beta = 0.0f;
            // A: [n, l, e]
            // B: [n, l, h]
            
            cl::Event event;
            int idx            = 0;
            cl_int ret = CL_SUCCESS;
            ret |= kernel->get().setArg(idx++, static_cast<int>(gemmSize[0]));
            ret |= kernel->get().setArg(idx++, static_cast<int>(gemmSize[1]));
            ret |= kernel->get().setArg(idx++, static_cast<int>(gemmSize[2]));
            ret |= kernel->get().setArg(idx++, alpha);
            ret |= kernel->get().setArg(idx++, beta);
            if(gemmSize[4] > 1) {
                int batch_offset_a = gemmSize[0] * gemmSize[2];
                int batch_offset_b = gemmSize[1] * gemmSize[2];
                int batch_offset_c = gemmSize[0] * gemmSize[1];

                ret |= kernel->get().setArg(idx++, tensorMemory[0]);
                ret |= kernel->get().setArg(idx++, batch_offset_a);
                ret |= kernel->get().setArg(idx++, tensorMemory[1]);
                ret |= kernel->get().setArg(idx++, batch_offset_b);
                if(gemmSize[5] == 1) {
                    ret |= kernel->get().setArg(idx++, tensorMemory[3]);
                    ret |= kernel->get().setArg(idx++, gemmSize[1]);
                }
                ret |= kernel->get().setArg(idx++, tensorMemory[2]);
                ret |= kernel->get().setArg(idx++, batch_offset_c);
                
                MNN_CHECK_CL_SUCCESS(ret, "setArg getGemmParams XgemmBatchhed Kernel");
                
                auto res = CL_SUCCESS;
                res = runtime->commandQueue().enqueueNDRangeKernel(kernel->get(), cl::NullRange, {globalWorkSize[0], globalWorkSize[1], globalWorkSize[2]}, {localWorkSize[0], localWorkSize[1], localWorkSize[2]}, nullptr, &event);
                if (res != CL_SUCCESS) {
                    MNN_PRINT("XgemmBatched params tune error: %d\n", res);
                    continue;
                }
            } else {
                int offset_a = 0;
                int offset_b = 0;
                int offset_c = 0;
                
                ret |= kernel->get().setArg(idx++, tensorMemory[0]);
                ret |= kernel->get().setArg(idx++, tensorMemory[1]);
                if(gemmSize[5] == 1) {
                    ret |= kernel->get().setArg(idx++, tensorMemory[3]);
                }
                ret |= kernel->get().setArg(idx++, tensorMemory[2]);
                ret |= kernel->get().setArg(idx++, offset_a);
                ret |= kernel->get().setArg(idx++, offset_b);
                ret |= kernel->get().setArg(idx++, offset_c);
                
                MNN_CHECK_CL_SUCCESS(ret, "setArg getGemmParams Xgemm Kernel");
                
                auto res = CL_SUCCESS;
                res = runtime->commandQueue().enqueueNDRangeKernel(kernel->get(), cl::NullRange, {globalWorkSize[0], globalWorkSize[1]}, {localWorkSize[0], localWorkSize[1]}, nullptr, &event);
                if (res != CL_SUCCESS) {
                    MNN_PRINT("Xgemm params tune error: %d\n", res);
                    continue;
                }
            }
            
            
            int cost_time = (int)runtime->getCostTime(&event);
            if(cost_time < min_cost) {
                min_cost = cost_time;
                params_prefer[0]  = kwg;
                params_prefer[1]  = kwi;
                params_prefer[2]  = mdima;
                params_prefer[3]  = mdimc;
                params_prefer[4]  = mwg;
                params_prefer[5]  = ndimb;
                params_prefer[6]  = ndimc;
                params_prefer[7]  = nwg;
                params_prefer[8]  = sa;
                params_prefer[9]  = sb;
                params_prefer[10] = strm;
                params_prefer[11] = strn;
                params_prefer[12] = vwm;
                params_prefer[13] = vwn;
                #ifdef TIME_TUNE_LOG
                for(auto &iter : params_prefer) {
                    MNN_PRINT("%d ", iter);
                }
                MNN_PRINT(": %d us, shape:%d %d %d batch:%d, layout:%d bias:%d, flops:%f GFLOPS\n", min_cost, gemmSize[0], gemmSize[1], gemmSize[2], gemmSize[4], gemmSize[3], gemmSize[5], 2.0 / 1000.0 * gemmSize[0] * gemmSize[1] * gemmSize[2] * gemmSize[4] / min_cost);
                #endif
            }
        }
    }
  
    if (tunedGemmParams.find(info) == tunedGemmParams.end()) {
        tunedGemmParams.insert(std::make_pair(info, params_prefer));
    }

    return params_prefer;
}

} // namespace OpenCL
} // namespace MNN
