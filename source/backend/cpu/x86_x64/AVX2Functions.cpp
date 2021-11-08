//
//  AVX2Functions.cpp
//  MNN
//
//  Created by MNN on b'2021/05/17'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "AVX2Functions.hpp"
#include "AVX2Backend.hpp"
#include "avx/FunctionSummary.hpp"
#include "avxfma/FunctionSummary.hpp"
#include "avx512/FunctionSummary.hpp"
namespace MNN {
struct MatMulPackParam {
    int eP;
    int lP;
    int hP;
};

static MatMulPackParam gPackInfo;
static CoreFunctions* gAVX2CoreFunctions = nullptr;
static CoreInt8Functions* gAVX2CoreInt8Functions = nullptr;
static void _MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = gPackInfo.eP;
    *lP = gPackInfo.lP;
    *hP = gPackInfo.hP;
}

bool AVX2Functions::init(int cpuFlags) {
    gAVX2CoreFunctions = new CoreFunctions;
    auto coreFunction = gAVX2CoreFunctions;
    gAVX2CoreInt8Functions = new CoreInt8Functions;
    // Init default functions
    *coreFunction = *MNNGetCoreFunctions();
    *gAVX2CoreInt8Functions = *MNNGetInt8CoreFunctions();
    _AVX_MNNInt8FunctionInit(gAVX2CoreInt8Functions);
    // Init AVX2
    coreFunction->MNNGetMatMulPackMode = _MNNGetMatMulPackMode;
    gPackInfo.eP                    = 24;
    gPackInfo.lP                    = 1;
    gPackInfo.hP                    = 4;
    _AVX_ReorderInit(coreFunction);

    coreFunction->MNNPackedMatMul       = _AVX_MNNPackedMatMul;
    coreFunction->MNNPackedMatMulRemain = _AVX_MNNPackedMatMulRemain;
    coreFunction->MNNPackC4ForMatMul_A  = _AVX_MNNPackC4ForMatMul_A;
    coreFunction->MNNPackForMatMul_B    = _AVX_MNNPackForMatMul_B;
    coreFunction->MNNComputeMatMulForE_1 = _AVX_MNNComputeMatMulForE_1;
    coreFunction->MNNComputeMatMulForH_1 = _AVX_MNNComputeMatMulForH_1;

    // For Packed Functions
    coreFunction->pack = 8;
    _AVX_ExtraInit(coreFunction);
    // Winograd
    _AVX_WinogradInit(coreFunction);
    if (cpuFlags & libyuv::kCpuHasFMA3) {
        coreFunction->MNNPackedMatMul       = _AVX_MNNPackedMatMulFMA;
        coreFunction->MNNPackedMatMulRemain = _AVX_MNNPackedMatMulRemainFMA;
        coreFunction->MNNComputeMatMulForE_1 = _AVX_MNNComputeMatMulForE_1FMA;
        coreFunction->MNNComputeMatMulForH_1 = _AVX_MNNComputeMatMulForH_1FMA;
        _AVX_ExtraInitFMA(coreFunction);
    }
#ifdef MNN_AVX512
    if ((cpuFlags & libyuv::kCpuHasAVX512VNNI)
        || (cpuFlags & libyuv::kCpuHasAVX512VL)
        || (cpuFlags & libyuv::kCpuHasAVX512BW)
        || (cpuFlags & libyuv::kCpuHasAVX512VBMI)
        || (cpuFlags & libyuv::kCpuHasAVX512VBITALG)
        || (cpuFlags & libyuv::kCpuHasAVX512VPOPCNTDQ)
        || (cpuFlags & libyuv::kCpuHasAVX512VBMI2)
        ) {
        coreFunction->pack = 16;
        _AVX512_ReorderInit(coreFunction);
        _AVX512_ExtraInit(coreFunction);
        _AVX512_WinogradInit(coreFunction);
        coreFunction->MNNPackForMatMul_B    = _AVX512_MNNPackForMatMul_B;
        coreFunction->MNNPackC4ForMatMul_A  = _AVX512_MNNPackC8ForMatMul_A;
        coreFunction->MNNPackedMatMul = _AVX512_MNNPackedMatMul;
        coreFunction->MNNPackedMatMulRemain = _AVX512_MNNPackedMatMulRemain;
        gPackInfo.eP                    = 48;
        gPackInfo.hP                    = 8;
        gPackInfo.lP                    = 1;
    }
#ifdef MNN_AVX512_VNNI
    if (cpuFlags & libyuv::kCpuHasAVX512VNNI) {
        _AVX512_MNNInt8FunctionInit(gAVX2CoreInt8Functions);
    }
#endif
#endif
    return true;
}
CoreFunctions* AVX2Functions::get() {
    return gAVX2CoreFunctions;
}
CoreInt8Functions* AVX2Functions::getInt8() {
    return gAVX2CoreInt8Functions;
}

};
