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
static void _MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = gPackInfo.eP;
    *lP = gPackInfo.lP;
    *hP = gPackInfo.hP;
}

bool AVX2Functions::init(int cpuFlags) {
    gAVX2CoreFunctions = new CoreFunctions;
    auto coreFunction = gAVX2CoreFunctions;
    // Init default functions
    *coreFunction = *MNNGetCoreFunctions();

    // Init AVX2
    coreFunction->MNNGetMatMulPackMode = _MNNGetMatMulPackMode;
    gPackInfo.eP                    = 24;
    gPackInfo.lP                    = 1;
    gPackInfo.hP                    = 4;
    coreFunction->pack = 8;
    coreFunction->MNNPackCUnit = _AVX_MNNPackCUnit;
    coreFunction->MNNUnpackCUnit = _AVX_MNNUnpackCUnit;
    coreFunction->MNNPackCUnitTranspose = _AVX_MNNPackCUnitTranspose;
    coreFunction->MNNUnpackCUnitTranspose = _AVX_MNNUnpackCUnitTranspose;
    coreFunction->MNNCopyC4WithStride = _AVX_MNNCopyC4WithStride;
    coreFunction->MNNAddC4WithStride = _AVX_MNNAddC4WithStride;
    coreFunction->MNNScaleAndAddBias = _AVX_MNNScaleAndAddBias;
    coreFunction->MNNMatrixAdd          = _AVX_MNNMatrixAdd;
    coreFunction->MNNMatrixSub          = _AVX_MNNMatrixSub;
    coreFunction->MNNPackedMatMul       = _AVX_MNNPackedMatMul;
    coreFunction->MNNPackedMatMulRemain = _AVX_MNNPackedMatMulRemain;
    coreFunction->MNNPackC4ForMatMul_A  = _AVX_MNNPackC4ForMatMul_A;
    coreFunction->MNNPackForMatMul_B    = _AVX_MNNPackForMatMul_B;
    coreFunction->MNNConvRunForUnitDepthWise = _AVX_MNNConvRunForUnitDepthWise;
    coreFunction->MNNConvRunForLineDepthwise = _AVX_MNNConvRunForLineDepthwise;
    coreFunction->MNNAxByClampBroadcastUnit = _AVX_MNNAxByClampBroadcastUnit;
    coreFunction->MNNComputeMatMulForE_1 = _AVX_MNNComputeMatMulForE_1;
    coreFunction->MNNComputeMatMulForH_1 = _AVX_MNNComputeMatMulForH_1;
    coreFunction->MNNStrassenMergeCFunction = _AVX_MNNStrassenMergeCFunction;
    coreFunction->MNNMultiAndDestTransformCommon23 = _AVX_MNNMultiAndDestTransformCommon23;
    coreFunction->MNNSourceTransformCommonF23 = _AVX_MNNSourceTransformCommonF23;
    coreFunction->MNNConvDwF23MulTransUnit = _AVX_MNNConvDwF23MulTransUnit;
    coreFunction->MNNReluWithSlopeChannel = _AVX_MNNReluWithSlopeChannel;
    coreFunction->MNNDeconvRunForLineDepthwise = _AVX_MNNDeconvRunForLineDepthwise;
    coreFunction->MNNDeconvRunForUnitDepthWise = _AVX_MNNDeconvRunForUnitDepthWise;
    coreFunction->MNNGridSampleInterp = _AVX_MNNGridSampleInterp;
    // For Pooling / Binary
    _AVX_ExtraInit(coreFunction);
    // Winograd
    _AVX_WinogradInit(coreFunction);
    if (cpuFlags & libyuv::kCpuHasFMA3) {
        coreFunction->MNNPackedMatMul       = _AVX_MNNPackedMatMulFMA;
        coreFunction->MNNPackedMatMulRemain = _AVX_MNNPackedMatMulRemainFMA;
        coreFunction->MNNComputeMatMulForE_1 = _AVX_MNNComputeMatMulForE_1FMA;
        coreFunction->MNNComputeMatMulForH_1 = _AVX_MNNComputeMatMulForH_1FMA;
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
        coreFunction->MNNPackForMatMul_B    = _AVX512_MNNPackForMatMul_B;
        coreFunction->MNNPackC4ForMatMul_A  = _AVX512_MNNPackC8ForMatMul_A;
        coreFunction->MNNPackedMatMul = _AVX512_MNNPackedMatMul;
        coreFunction->MNNPackedMatMulRemain = _AVX512_MNNPackedMatMulRemain;
        gPackInfo.eP                    = 48;
        gPackInfo.hP                    = 8;
        gPackInfo.lP                    = 1;
    }
#endif
    return true;
}
CoreFunctions* AVX2Functions::get() {
    return gAVX2CoreFunctions;
}
};
