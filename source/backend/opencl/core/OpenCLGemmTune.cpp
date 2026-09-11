//
//  OpenCLGemmTune.cpp
//  MNN
//
//  Created by MNN on 2024/05/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "backend/opencl/core/OpenCLTuneHeuristic.hpp"
#include <algorithm>
#include <string>
#include <math.h>
#include <set>
#include <vector>
#include "core/Macro.h"

namespace MNN {
namespace OpenCL {

static void generateCombinations(const std::vector<std::vector<uint32_t>>& candidates,
                                 std::vector<uint32_t>& currentCombination,
                                 std::vector<std::vector<uint32_t>>& totalCombinations, int depth) {
    if (depth == candidates.size()) {
        totalCombinations.emplace_back(currentCombination);
        return;
    }

    for (int i = 0; i < candidates[depth].size(); i++) {
        currentCombination[depth] = candidates[depth][i];
        generateCombinations(candidates, currentCombination, totalCombinations, depth + 1);
    }
}

static bool isCandidateValid(uint32_t kwg, uint32_t kwi, uint32_t mwg, uint32_t mdimc, uint32_t vwm, uint32_t nwg,
                             uint32_t ndimc, uint32_t vwn, uint32_t mdima, uint32_t ndimb, uint32_t sa, uint32_t sb,
                             OpenCLRuntime* runtime, const std::vector<uint32_t>& gemmSize, int precision) {
    // problem size align
    if (gemmSize[0] % mwg != 0 || gemmSize[1] % nwg != 0) {
        return false;
    }

    if (mwg % (mdimc * vwm) != 0 || mwg % (mdima * vwm) != 0) {
        return false;
    }
    if (nwg % (ndimc * vwn) != 0 || nwg % (ndimb * vwn) != 0) {
        return false;
    }
    uint32_t kdima = (mdimc * ndimc) / mdima;
    uint32_t kdimb = (mdimc * ndimc) / ndimb;
    if (sa == 1 || sb == 1) {
        // params align
        if (kwg % kwi != 0) {
            return false;
        }
        if (kwg % kdima != 0 || kwg % kdimb != 0) {
            return false;
        }
        if (gemmSize[2] % kwg != 0) {
            return false;
        }
    }

    if (mdimc != mdima || ndimc != ndimb) {
        return false;
    }
    if (sa != sb) {
        return false;
    }

    // no local memory no need tune kwg
    if (sa == 0 && sb == 0 && kwg == 32) {
        return false;
    }

    // local memory limit
    uint32_t local_mem_size = 0;
    if (sa) {
        local_mem_size += kwg * mwg;
    }
    if (sb) {
        local_mem_size += kwg * nwg;
    }
    if (precision != BackendConfig::Precision_High) {
        local_mem_size *= 2;
    } else {
        local_mem_size *= 4;
    }
    if (local_mem_size > runtime->getMaxLocalMem()) {
        return false;
    }

    // local size limit
    if (mdimc * ndimc > runtime->MaxWorkGroupSize()) {
        return false;
    }

    bool totalLarge = 1.0 * gemmSize[0] / 1024 * gemmSize[1] / 1024 * gemmSize[2] / 1024 >= 0.5;
    bool dimLarge = gemmSize[0] > 128 && gemmSize[1] > 128 && gemmSize[2] > 128;
    if (gemmSize[4] == 1) {
        if (totalLarge && dimLarge) {
            if (mwg * nwg < 128 * 64) {
                return false;
            }
            if (mdimc * ndimc < 16 * 8) {
                return false;
            }
            if (vwm * vwn < 4 * 4) {
                return false;
            }
        } else {
            if (mwg * nwg > 128 * 64) {
                return false;
            }
            if (mdimc * ndimc > 16 * 8) {
                return false;
            }
            if (vwm * vwn > 4 * 4) {
                return false;
            }
        }
    }

    return true;
}

static bool GemmlocalWSTune(const std::map<std::string, std::vector<TuneInfo>>& tuneMap,
                            const std::vector<uint32_t>& gemmSize, std::vector<uint32_t>& res, OpenCLRuntime* runtime,
                            int precision) {
    auto iter = tuneMap.find("Xgemm_tune");
    if (iter == tuneMap.end()) {
        return false;
    }
    auto TuneInfoVec = iter->second;
    uint32_t minPoint = UINT_MAX;
    int index = -1;
    for (int i = 0; i < TuneInfoVec.size(); ++i) {
        // Layout+Precision, Batch, Bias+GroupSize must equall
        if (gemmSize[3] != TuneInfoVec[i].globalSize[3] || gemmSize[4] != TuneInfoVec[i].globalSize[4] ||
            gemmSize[5] != TuneInfoVec[i].globalSize[5]) {
            continue;
        }
        auto combinations = TuneInfoVec[i].localSize;
        uint32_t kwg = combinations[0];
        uint32_t kwi = combinations[1];
        uint32_t mdima = combinations[2];
        uint32_t mdimc = combinations[3];
        uint32_t mwg = combinations[4];
        uint32_t ndimb = combinations[5];
        uint32_t ndimc = combinations[6];
        uint32_t nwg = combinations[7];
        uint32_t sa = combinations[8];
        uint32_t sb = combinations[9];
        uint32_t strm = combinations[10];
        uint32_t strn = combinations[11];
        uint32_t vwm = combinations[12];
        uint32_t vwn = combinations[13];

        if (!isCandidateValid(kwg, kwi, mwg, mdimc, vwm, nwg, ndimc, vwn, mdima, ndimb, sa, sb, runtime, gemmSize,
                              precision)) {
            continue;
        }
        uint32_t point = 0;
        for (int j = 0; j < 3; ++j) {
            point += std::abs(static_cast<int>(gemmSize[j]) - static_cast<int>(TuneInfoVec[i].globalSize[j]));
        }

        if (point < minPoint) {
            index = i;
            minPoint = point;
        }
    }
    if (index != -1) {
        res = TuneInfoVec[index].localSize;
    } else {
        return false;
    }
    return true;
}

// Build the -D option set for one Xgemm candidate. Shared by tuning, prebuild, and execution so
// every path produces byte-identical program cache keys.
std::set<std::string> makeGemmBuildOptions(const std::vector<uint32_t>& params, int layoutType, int biasType,
                                           int mixPrecision, GpuType gpuType) {
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DKWG=" + std::to_string(params[0]));
    buildOptions.emplace("-DKWI=" + std::to_string(params[1]));
    buildOptions.emplace("-DMDIMA=" + std::to_string(params[2]));
    buildOptions.emplace("-DMDIMC=" + std::to_string(params[3]));
    buildOptions.emplace("-DMWG=" + std::to_string(params[4]));
    buildOptions.emplace("-DNDIMB=" + std::to_string(params[5]));
    buildOptions.emplace("-DNDIMC=" + std::to_string(params[6]));
    buildOptions.emplace("-DNWG=" + std::to_string(params[7]));
    buildOptions.emplace("-DSA=" + std::to_string(params[8]));
    buildOptions.emplace("-DSB=" + std::to_string(params[9]));
    buildOptions.emplace("-DSTRM=" + std::to_string(params[10]));
    buildOptions.emplace("-DSTRN=" + std::to_string(params[11]));
    buildOptions.emplace("-DVWM=" + std::to_string(params[12]));
    buildOptions.emplace("-DVWN=" + std::to_string(params[13]));

    if (layoutType >= 4) {
        buildOptions.emplace("-DOUTPUTMN");
    }
    if (gpuType == GpuType::ADRENO) {
        buildOptions.emplace("-DUSE_CL_MAD=1");
        buildOptions.emplace("-DRELAX_WORKGROUP_SIZE=1");
    }

    if (biasType >= 1) {
        buildOptions.emplace("-DBIAS_TYPE=" + std::to_string(biasType));
    }
    if (mixPrecision > 0) {
        buildOptions.emplace("-DPRECISION_COMPUTE=float -DCONVERT_PRECISION_COMPUTE=convert_float");
        buildOptions.emplace("-DPRECISION_COMPUTE2=float2 -DCONVERT_PRECISION_COMPUTE2=convert_float2");
        buildOptions.emplace("-DPRECISION_COMPUTE4=float4 -DCONVERT_PRECISION_COMPUTE4=convert_float4");
        buildOptions.emplace("-DPRECISION_COMPUTE8=float8 -DCONVERT_PRECISION_COMPUTE8=convert_float8");
        buildOptions.emplace("-DPRECISION_COMPUTE16=float16 -DCONVERT_PRECISION_COMPUTE16=convert_float16");
    }
    return buildOptions;
}

static const std::vector<std::vector<uint32_t>>& wideXgemmPool() {
    static const std::vector<std::vector<uint32_t>> pool = {
        {16, 2, 16, 16, 64, 8, 8, 128, 0, 0, 0, 0, 4, 8},    {16, 2, 16, 16, 128, 8, 8, 64, 0, 0, 0, 0, 8, 8},
        {16, 2, 16, 16, 128, 16, 16, 128, 0, 0, 0, 0, 8, 8}, {16, 2, 16, 16, 128, 8, 8, 32, 0, 0, 0, 1, 8, 4},
        {16, 2, 8, 8, 16, 8, 8, 64, 0, 0, 0, 0, 2, 8},       {16, 2, 16, 16, 64, 8, 8, 128, 0, 0, 0, 1, 4, 8},
        {16, 2, 8, 8, 32, 8, 8, 128, 0, 0, 1, 0, 2, 8},      {16, 2, 16, 16, 64, 8, 8, 128, 0, 0, 1, 1, 2, 8},
        {16, 2, 16, 16, 128, 8, 8, 64, 0, 0, 1, 1, 2, 8},    {16, 2, 16, 16, 128, 8, 8, 128, 0, 0, 0, 0, 8, 8},
        {16, 2, 8, 8, 16, 8, 8, 128, 0, 0, 0, 0, 2, 8},      {16, 2, 4, 4, 32, 8, 8, 32, 0, 0, 0, 0, 8, 2},
        {16, 2, 4, 4, 16, 8, 8, 32, 0, 0, 0, 0, 4, 2},       {16, 2, 16, 16, 64, 8, 8, 128, 0, 0, 1, 0, 2, 8},
    };
    return pool;
}

static uint32_t getMaxGemmDivisor(uint32_t value) {
    const uint32_t divisors[] = {128, 64, 32};
    for (auto divisor : divisors) {
        if (value % divisor == 0) {
            return divisor;
        }
    }
    return 16;
}

static std::vector<uint32_t> makeGemmParamsPreference(const std::vector<uint32_t>& gemmSize) {
    std::vector<uint32_t> params = {16, 2, 4, 4, 16, 4, 4, 16, 0, 0, 1, 0, 2, 2};
    const auto threadRatio = 1.0 * gemmSize[4] * gemmSize[0] / 512.0 * gemmSize[1] / 512.0;
    if (threadRatio >= 1.0 && gemmSize[0] % 64 == 0 && gemmSize[1] % 32 == 0) {
        params.assign({16, 2, 16, 16, 64, 8, 8, 32, 0, 0, 0, 0, 4, 4});
    }
    return params;
}

static bool getPresetGemmParams(const std::vector<uint32_t>& gemmSize, OpenCLRuntime* runtime, int precision,
                                int tuneLevel, std::vector<uint32_t>& params) {
    auto info = makeGemmTuneInfoKey(gemmSize, precision);
    {
        std::lock_guard<std::mutex> gemmLock(runtime->gemmParamsMutex());
        auto& tunedGemmParams = runtime->tunedGemmParamsMap();
        auto cached = tunedGemmParams.find(info);
        if (cached != tunedGemmParams.end()) {
            params = cached->second;
            return true;
        }
    }

    if (runtime->getGpuLevel() >= MEDIUM) {
        const auto computeRatio = 1.0 * gemmSize[4] * gemmSize[0] / 256.0 * gemmSize[1] / 256.0 * gemmSize[2] / 256.0;
        const auto threadRatio = 1.0 * gemmSize[4] * gemmSize[0] / 256.0 * gemmSize[1] / 256.0;
        const bool isEven = (gemmSize[0] >= 256 && gemmSize[1] >= 128 && gemmSize[2] >= 128) ||
                            (gemmSize[1] >= 128 && gemmSize[2] >= 128 && gemmSize[4] >= 4);
        if (computeRatio >= 1.0 && threadRatio >= 1.0 && isEven && gemmSize[0] % 64 == 0 && gemmSize[1] % 32 == 0) {
            const auto divisorM = std::min<uint32_t>(getMaxGemmDivisor(gemmSize[0]), 64);
            const auto divisorN = std::min<uint32_t>(getMaxGemmDivisor(gemmSize[1]), 32);
            params = {16, 2, divisorM / 4, divisorM / 4, divisorM, divisorN / 4, divisorN / 4, divisorN, 0, 0, 0, 0,
                      4,  4};
            return true;
        }
    }
    if (runtime->getGpuLevel() == TOP && (tuneLevel == None || tuneLevel == Fast)) {
        const auto computeRatio = 1.0 * gemmSize[4] * gemmSize[0] / 512.0 * gemmSize[1] / 512.0 * gemmSize[2] / 512.0;
        const auto threadRatio = 1.0 * gemmSize[4] * gemmSize[0] / 512.0 * gemmSize[1] / 512.0;
        const bool isEven = (gemmSize[0] >= 512 && gemmSize[1] >= 256 && gemmSize[2] >= 256) ||
                            (gemmSize[1] >= 128 && gemmSize[2] >= 128 && gemmSize[4] >= 4);
        if (computeRatio >= 1.0 && threadRatio >= 1.0 && isEven && gemmSize[0] % 64 == 0 && gemmSize[1] % 64 == 0) {
            const auto divisorM = getMaxGemmDivisor(gemmSize[0]);
            const auto divisorN = getMaxGemmDivisor(gemmSize[1]);
            params = {16, 2, 16, 16, divisorM, 16, 16, divisorN, 0, 0, 0, 0, divisorM / 16, divisorN / 16};
            return true;
        }
    }
    return GemmlocalWSTune(runtime->getTuneLwsMap(), gemmSize, params, runtime, precision);
}

static std::vector<uint32_t> getImmediateGemmParams(const std::vector<uint32_t>& gemmSize,
                                                    const std::vector<uint32_t>& paramsPreference,
                                                    OpenCLRuntime* runtime, int precision) {
    auto params = getHeuristicXgemmParams(gemmSize[0], gemmSize[1], gemmSize[2], gemmSize[4], runtime->getGpuType(),
                                          runtime->getGpuLevel());
    if (params.size() == 14 &&
        isCandidateValid(params[0], params[1], params[4], params[3], params[12], params[7], params[6], params[13],
                         params[2], params[5], params[8], params[9], runtime, gemmSize, precision)) {
        return params;
    }

    const float multiNum = 1.0 * gemmSize[0] / 512.0 * gemmSize[1] / 512.0 * gemmSize[2] / 512.0;
    const auto divisorM = getMaxGemmDivisor(gemmSize[0]);
    const auto divisorN = getMaxGemmDivisor(gemmSize[1]);
    if (gemmSize[4] == 1) {
        if (gemmSize[0] >= 256 && gemmSize[1] >= 256 && gemmSize[2] >= 256) {
            if (multiNum > 8.0 && divisorM >= 128 && divisorN >= 64) {
                return {16, 2, 16, 16, 128, 8, 8, 64, 0, 0, 0, 1, 8, 8};
            }
            if (divisorM >= 64 && divisorN >= 64) {
                return {16, 2, 8, 8, 64, 8, 8, 64, 0, 0, 0, 1, 8, 8};
            }
        }
    } else if (divisorM >= 64 && divisorN >= 128) {
        return {16, 2, 16, 16, 64, 8, 8, 128, 0, 0, 1, 0, 2, 8};
    } else if (divisorM >= 64 && divisorN >= 64) {
        return {16, 2, 8, 8, 64, 8, 8, 64, 0, 0, 1, 0, 4, 4};
    }
    return paramsPreference;
}

// Enumerate, validate, dedup and prune the candidate pool for a shape. params_prefer (index 0)
// always survives pruning.
static std::vector<std::vector<uint32_t>> buildGemmCandidateList(const std::vector<uint32_t>& gemmSize,
                                                                 const std::vector<uint32_t>& params_prefer,
                                                                 int tuneLevel, OpenCLRuntime* runtime, int precision) {
    std::vector<std::vector<uint32_t>> totalCombinations;
    totalCombinations.emplace_back(params_prefer);
    // Cap candidates actually measured; oversized Heavy pools are scored and pruned below.
    size_t maxMeasure = 16;

    if (tuneLevel >= Wide) {
        const auto& widePool = wideXgemmPool();
        totalCombinations.insert(totalCombinations.end(), widePool.begin(), widePool.end());
        if (runtime->getGpuType() == MALI) {
            totalCombinations.push_back({16, 2, 16, 16, 128, 8, 8, 64, 0, 0, 1, 0, 8, 8});
            totalCombinations.push_back({16, 2, 16, 16, 128, 8, 8, 64, 0, 0, 0, 1, 8, 8});
            totalCombinations.push_back({16, 2, 16, 16, 128, 16, 16, 128, 0, 0, 0, 1, 8, 8});
        }
        maxMeasure = totalCombinations.size();
    } else {
        // get all combinations
        std::vector<std::vector<uint32_t>> candidates = {
            {16, 32},          // KWG
            {2},               // KWI
            {4, 8, 16},        // MDIMA
            {4, 8, 16},        // MDIMC
            {16, 32, 64, 128}, // MWG
            {8, 16},           // NDIMB
            {8, 16},           // NDIMC
            {16, 32, 64, 128}, // NWG
            {0},               // SA
            {0},               // SB
            {0, 1},            // STRM
            {0, 1},            // STRN
            {2, 4, 8},         // VWM
            {2, 4, 8}          // VWN
        };

        std::vector<uint32_t> currentCombination(candidates.size());
        generateCombinations(candidates, currentCombination, totalCombinations, 0);
    }
    // Dedup and filter first, then prune oversized pools: every candidate costs one kernel
    // build (unique -D set) plus one measured run, so only candidates close to the heuristic
    // preference are worth measuring. params_prefer (index 0) always stays in the pool.
    std::vector<std::vector<uint32_t>> candidates;
    {
        std::set<std::vector<uint32_t>> seen;
        for (auto& combo : totalCombinations) {
            if (combo.size() != 14 || !seen.insert(combo).second) {
                continue;
            }
            if (!isCandidateValid(combo[0], combo[1], combo[4], combo[3], combo[12], combo[7], combo[6], combo[13],
                                  combo[2], combo[5], combo[8], combo[9], runtime, gemmSize, precision)) {
                continue;
            }
            candidates.push_back(combo);
        }
    }
    if (candidates.size() > maxMeasure) {
        auto refParams = getHeuristicXgemmParams(gemmSize[0], gemmSize[1], gemmSize[2], gemmSize[4],
                                                 runtime->getGpuType(), runtime->getGpuLevel());
        auto distance = [&refParams](const std::vector<uint32_t>& combo) -> float {
            if (refParams.size() != 14) {
                return 0.f; // no reference for this device: keep pool order
            }
            // tile-shape dims dominate kernel behavior; STRM/STRN matter on Adreno
            const int dims[] = {2, 3, 4, 5, 6, 7, 10, 11, 12, 13};
            float score = 0.f;
            for (int d : dims) {
                float a = static_cast<float>(combo[d]);
                float b = static_cast<float>(refParams[d]);
                score += fabsf(a - b) / std::max(std::max(a, b), 1.f);
            }
            return score;
        };
        std::vector<std::pair<float, size_t>> order;
        order.reserve(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            // negative score keeps params_prefer regardless of the reference
            order.emplace_back(i == 0 ? -1.f : distance(candidates[i]), i);
        }
        std::stable_sort(order.begin(), order.end());
        std::vector<std::vector<uint32_t>> pruned;
        pruned.reserve(maxMeasure);
        for (size_t i = 0; i < maxMeasure; ++i) {
            pruned.push_back(candidates[order[i].second]);
        }
        candidates.swap(pruned);
    }
    return candidates;
}

std::vector<std::set<std::string>> getGemmPrebuildOptions(const std::vector<uint32_t>& gemmSize, int precision,
                                                          int tuneLevel, OpenCLRuntime* runtime) {
    MNN_ASSERT(gemmSize.size() == 6);
    std::vector<std::set<std::string>> result;
    std::set<std::set<std::string>> uniqueOptions;
    const int layoutType = gemmSize[3] % 10;
    const int mixPrecision = gemmSize[3] / 10;
    const int biasType = gemmSize[5] % 10;
    auto append = [&](const std::vector<uint32_t>& params) {
        auto options = makeGemmBuildOptions(params, layoutType, biasType, mixPrecision, runtime->getGpuType());
        if (uniqueOptions.insert(options).second) {
            result.emplace_back(std::move(options));
        }
    };

    std::vector<uint32_t> preset;
    if (getPresetGemmParams(gemmSize, runtime, precision, tuneLevel, preset)) {
        append(preset);
        return result;
    }

    auto paramsPreference = makeGemmParamsPreference(gemmSize);
    if (tuneLevel == None || tuneLevel == Fast) {
        append(getImmediateGemmParams(gemmSize, paramsPreference, runtime, precision));
        return result;
    }

    auto candidates = buildGemmCandidateList(gemmSize, paramsPreference, tuneLevel, runtime, precision);
    for (const auto& candidate : candidates) {
        append(candidate);
    }
    append(getImmediateGemmParams(gemmSize, paramsPreference, runtime, precision));
    return result;
}

// Compute gws/lws from a candidate's params, bind args and enqueue on `queue`, recording `event`.
// Returns the enqueue result (CL_SUCCESS on success). The caller must already have verified that
// mdimc*ndimc fits the kernel's max work-group size.
static cl_int enqueueGemmCandidate(cl::Kernel& kernel, cl::CommandQueue& queue, const std::vector<uint32_t>& params,
                                   const std::vector<uint32_t>& gemmSize, int layoutType, int biasType, int groupSize,
                                   const std::vector<cl::Buffer>& tensorMemory, cl::Event& event) {
    int localM = params[3];                    // mdimc
    int localN = params[6];                    // ndimc
    int out_per_thread_m = params[4] / localM; // mwg / localM
    int out_per_thread_n = params[7] / localN; // nwg / localN

    std::vector<uint32_t> globalWorkSize = {static_cast<uint32_t>(gemmSize[0] / out_per_thread_m),
                                            static_cast<uint32_t>(gemmSize[1] / out_per_thread_n), gemmSize[4]};
    std::vector<uint32_t> localWorkSize = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN), 1};

    float alpha = 1.0;
    float beta = 0.0f;
    // A: [n, l, e]
    // B: [n, l, h]

    int idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= kernel.setArg(idx++, static_cast<int>(gemmSize[0]));
    ret |= kernel.setArg(idx++, static_cast<int>(gemmSize[1]));
    ret |= kernel.setArg(idx++, static_cast<int>(gemmSize[2]));
    ret |= kernel.setArg(idx++, alpha);
    ret |= kernel.setArg(idx++, beta);

    int stride[4] = {(int)gemmSize[0], (int)gemmSize[1], (int)gemmSize[1], (int)gemmSize[1]};
    if (layoutType < 4) {
        stride[2] = gemmSize[0]; // output: [N, M]
    }
    if (gemmSize[4] > 1) {
        int batch_offset_a = gemmSize[0] * gemmSize[2];
        int batch_offset_b = gemmSize[1] * gemmSize[2];
        int batch_offset_c = gemmSize[0] * gemmSize[1];
        int batch_offset[4] = {batch_offset_a, batch_offset_b, batch_offset_c, 0};
        int base_ptr_offset[4] = {0, 0, 0, 0};

        int group[4] = {1, (int)groupSize, 1, (int)gemmSize[4]};

        ret |= kernel.setArg(idx++, tensorMemory[0]);
        ret |= kernel.setArg(idx++, tensorMemory[1]);
        if (biasType > 0) {
            ret |= kernel.setArg(idx++, tensorMemory[3]);
        }
        ret |= kernel.setArg(idx++, tensorMemory[2]);
        ret |= kernel.setArg(idx++, sizeof(batch_offset), batch_offset);
        ret |= kernel.setArg(idx++, sizeof(batch_offset), base_ptr_offset);
        ret |= kernel.setArg(idx++, sizeof(stride), stride);
        ret |= kernel.setArg(idx++, sizeof(group), group);

        MNN_CHECK_CL_SUCCESS(ret, "setArg getGemmParams XgemmBatchhed Kernel");

        auto res =
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, {globalWorkSize[0], globalWorkSize[1], globalWorkSize[2]},
                                       {localWorkSize[0], localWorkSize[1], localWorkSize[2]}, nullptr, &event);
        if (res != CL_SUCCESS) {
            MNN_PRINT("XgemmBatched params tune error: %d\n", res);
        }
        return res;
    } else {
        int offset[4] = {0, 0, 0, 0};

        ret |= kernel.setArg(idx++, tensorMemory[0]);
        ret |= kernel.setArg(idx++, tensorMemory[1]);
        if (biasType >= 1) {
            ret |= kernel.setArg(idx++, tensorMemory[3]);
        }
        ret |= kernel.setArg(idx++, tensorMemory[2]);
        ret |= kernel.setArg(idx++, offset);
        ret |= kernel.setArg(idx++, stride);

        MNN_CHECK_CL_SUCCESS(ret, "setArg getGemmParams Xgemm Kernel");

        auto res = queue.enqueueNDRangeKernel(kernel, cl::NullRange, {globalWorkSize[0], globalWorkSize[1]},
                                              {localWorkSize[0], localWorkSize[1]}, nullptr, &event);
        if (res != CL_SUCCESS) {
            MNN_PRINT("Xgemm params tune error: %d\n", res);
        }
        return res;
    }
}

std::vector<uint32_t> makeGemmTuneInfoKey(const std::vector<uint32_t>& gemmSize, int precision) {
    MNN_ASSERT(gemmSize.size() == 6); // M, N, K, Layout+Precision, Batch, Bias+GroupSize
    int mixPrecision = gemmSize[3] / 10;
    std::vector<uint32_t> info(gemmSize);
    uint32_t precisionType = precision;
    if (precisionType == 2 && mixPrecision > 0) {
        precisionType = 0;
    }
    info.emplace_back(precisionType);
    return info;
}

std::vector<uint32_t> getGemmParams(const std::vector<uint32_t>& gemmSize, OpenCLRuntime* runtime, int precision,
                                    int tuneLevel) {
    MNN_ASSERT(gemmSize.size() == 6); // M, N, K, Layout+Precision, Batch, Bias+GroupSize
    MNN_ASSERT(gemmSize[0] % 16 == 0);
    MNN_ASSERT(gemmSize[1] % 16 == 0);
    MNN_ASSERT(gemmSize[2] % 4 == 0);

    std::vector<uint32_t> preset;
    if (getPresetGemmParams(gemmSize, runtime, precision, tuneLevel, preset)) {
        return preset;
    }

    auto paramsPreference = makeGemmParamsPreference(gemmSize);
    if (tuneLevel != None && tuneLevel != Fast && runtime->reserveGemmTuneSlot(gemmSize, precision)) {
        auto prepared = prepareGemmTuneCandidates(gemmSize, precision, tuneLevel, runtime, paramsPreference);
        runtime->submitGemmTuneJob(gemmSize, precision, std::move(prepared));
    }
    return getImmediateGemmParams(gemmSize, paramsPreference, runtime, precision);
}

std::vector<GemmTuneCandidate> prepareGemmTuneCandidates(const std::vector<uint32_t>& gemmSize, int precision,
                                                         int tuneLevel, OpenCLRuntime* runtime,
                                                         const std::vector<uint32_t>& params_prefer) {
    std::vector<GemmTuneCandidate> prepared;
    int layoutType = gemmSize[3] % 10;
    int mixPrecision = gemmSize[3] / 10;
    int biasType = gemmSize[5] % 10;
    auto candidates = buildGemmCandidateList(gemmSize, params_prefer, tuneLevel, runtime, precision);
    const char* kernelName = gemmSize[4] > 1 ? "XgemmBatched" : "Xgemm";
    prepared.reserve(candidates.size());
    for (auto& combo : candidates) {
        auto buildOptions = makeGemmBuildOptions(combo, layoutType, biasType, mixPrecision, runtime->getGpuType());
        // Foreground compile populates the shared in-process mBuildProgramMap: when the runtime
        // later creates the winning kernel via buildKernelWithCache it reuses the cached program
        // instead of rebuilding from source, and the worker doesn't have to build anything.
        cl::Program program = runtime->buildTuneProgram("matmul_params_buf", buildOptions, precision);
        if (program.get() == nullptr) {
            continue;
        }
        GemmTuneCandidate cand;
        cand.program = program;
        cand.kernelName = kernelName;
        cand.params = combo;
        prepared.push_back(std::move(cand));
    }
    return prepared;
}

std::vector<uint32_t> measureGemmTuneCandidates(const std::vector<GemmTuneCandidate>& candidates,
                                                const std::vector<uint32_t>& gemmSize, cl::Device& device,
                                                cl::CommandQueue& queue, const std::vector<cl::Buffer>& tensorMemory) {
    std::vector<uint32_t> best;
    if (candidates.empty()) {
        return best;
    }
    int layoutType = gemmSize[3] % 10;
    int biasType = gemmSize[5] % 10;
    int groupSize = gemmSize[5] / 10 + 1;

    // Create one kernel per candidate from its precompiled (immutable) program. clCreateKernel on
    // a built program is thread-safe, and these kernel objects plus `queue` are private to the
    // worker, so measuring here never races the foreground's use of the shared runtime.
    std::vector<std::unique_ptr<cl::Kernel>> kernels(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        cl_int kres = CL_SUCCESS;
        std::unique_ptr<cl::Kernel> kernel(
            new cl::Kernel(candidates[i].program, candidates[i].kernelName.c_str(), &kres));
        if (kres != CL_SUCCESS) {
            continue;
        }
        size_t maxWorkGroupSize = 0;
        kernel->getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &maxWorkGroupSize);
        int localM = candidates[i].params[3];
        int localN = candidates[i].params[6];
        if (static_cast<size_t>(localM) * static_cast<size_t>(localN) > maxWorkGroupSize) {
            continue; // this tile can't run on this device
        }
        kernels[i] = std::move(kernel);
    }

    // Enqueue every candidate back-to-back, then finish the queue once. One finish() is cheaper
    // than waiting for each event separately.
    std::vector<cl::Event> events(candidates.size());
    std::vector<bool> dispatched(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (kernels[i] == nullptr) {
            continue;
        }
        if (enqueueGemmCandidate(*kernels[i], queue, candidates[i].params, gemmSize, layoutType, biasType, groupSize,
                                 tensorMemory, events[i]) == CL_SUCCESS) {
            dispatched[i] = true;
        }
    }
    cl_int wres = queue.finish();
    MNN_CHECK_CL_SUCCESS(wres, "xgemm async tune");

    uint32_t minCost = UINT_MAX;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (!dispatched[i]) {
            continue;
        }
        auto startNanos = events[i].getProfilingInfo<CL_PROFILING_COMMAND_START>();
        auto stopNanos = events[i].getProfilingInfo<CL_PROFILING_COMMAND_END>();
        uint32_t cost = static_cast<uint32_t>((stopNanos - startNanos) / 1000);
        if (cost > 0 && cost < minCost) {
            minCost = cost;
            best = candidates[i].params;
        }
    }
    return best;
}

} // namespace OpenCL
} // namespace MNN
