//
//  RVVFunctions.cpp
//  MNN
//
//  Created by ihb2032 on 2026/04/20.
//  Email: hebome@foxmail.com
//
#include "RVVFunctions.hpp"

void MNNPackC4RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void MNNUnpackC4RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void MNNPackC4ForMatMul_ARVV(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el);
void MNNConvRunForLineDepthwiseRVV(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                   size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                   size_t srcHStep, size_t dstHStep, const float* bias, const float* parameters);
void MNNAxByClampBroadcastUnitRVV(float* C, const float* A, const float* B, size_t width, size_t cStride,
                                  size_t aStride, size_t height, const float* parameters);
void MNNMatrixAddRVV(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                     size_t bStride, size_t height);
void MNNMatrixSubRVV(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                     size_t bStride, size_t height);
void MNNStrassenMergeCFunctionRVV(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride,
                                  size_t eSub, size_t hSub);
void MNNScaleAndAddBiasRVV(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                           size_t biasNumber);
void MNNCopyC4WithStrideRVV(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);
void MNNAddC4WithStrideRVV(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);
void MNNDeconvRunForUnitDepthWiseRVV(const float* dst, float* src, const float* weight, size_t fw, size_t fh,
                                     size_t weight_y_step, size_t dilateX_step, size_t dilateY_step);
void MNNReluWithSlopeChannelRVV(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad);
void MNNRGBAToBGRARVV(const unsigned char* source, unsigned char* dest, size_t count);
void MNNNV21ToRGBARVV(const unsigned char* source, unsigned char* dest, size_t count);
void MNNNV21ToRGBRVV(const unsigned char* source, unsigned char* dest, size_t count);
void MNNNV21ToBGRARVV(const unsigned char* source, unsigned char* dest, size_t count);
void MNNNV21ToBGRRVV(const unsigned char* source, unsigned char* dest, size_t count);
void MNNSoftmaxRVV(float* softmaxDst, const float* input, float* runningMax, float* runningSum, float* updateScale,
                   int outside, int reduceSize, int kvSeqOffset, int validOffset, int pack, bool mask);

void MNNGemmInt8AddBiasScale_16x4_Unit_RVV(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad,
                                           size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post,
                                           size_t realCount);
void MNNGemmInt8AddBiasScale_16x4_w4_Unit_RVV(int8_t* dst, const int8_t* src, const int8_t* weight,
                                              size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                              const QuanPostTreatParameters* post, size_t realCount);

namespace MNN {

static CoreFunctions* gInstance = nullptr;
static CoreInt8Functions* gInt8Instance = nullptr;

void registerRVVBackend() {
    RVVFunctions::init();
}

bool RVVFunctions::init() {
    if (gInstance != nullptr) {
        return true;
    }
    auto core = MNNGetCoreFunctions();
    auto int8Core = MNNGetInt8CoreFunctions();
    if (core == nullptr || int8Core == nullptr || !core->supportRVV) {
        return false;
    }
    gInstance = new CoreFunctions;
    gInt8Instance = new CoreInt8Functions;
    *gInstance = *core;
    *gInt8Instance = *int8Core;

    gInstance->MNNPackCUnit = ::MNNPackC4RVV;
    gInstance->MNNUnpackCUnit = ::MNNUnpackC4RVV;
    gInstance->MNNPackC4ForMatMul_A = ::MNNPackC4ForMatMul_ARVV;
    gInstance->MNNConvRunForLineDepthwise = ::MNNConvRunForLineDepthwiseRVV;
    gInstance->MNNAxByClampBroadcastUnit = ::MNNAxByClampBroadcastUnitRVV;
    gInstance->MNNMatrixAdd = ::MNNMatrixAddRVV;
    gInstance->MNNMatrixSub = ::MNNMatrixSubRVV;
    gInstance->MNNStrassenMergeCFunction = ::MNNStrassenMergeCFunctionRVV;
    gInstance->MNNScaleAndAddBias = ::MNNScaleAndAddBiasRVV;
    gInstance->MNNCopyC4WithStride = ::MNNCopyC4WithStrideRVV;
    gInstance->MNNAddC4WithStride = ::MNNAddC4WithStrideRVV;
    gInstance->MNNDeconvRunForUnitDepthWise = ::MNNDeconvRunForUnitDepthWiseRVV;
    gInstance->MNNReluWithSlopeChannel = ::MNNReluWithSlopeChannelRVV;
    gInstance->MNNRGBAToBGRA = ::MNNRGBAToBGRARVV;
    gInstance->MNNNV21ToRGBA = ::MNNNV21ToRGBARVV;
    gInstance->MNNNV21ToRGB = ::MNNNV21ToRGBRVV;
    gInstance->MNNNV21ToBGRA = ::MNNNV21ToBGRARVV;
    gInstance->MNNNV21ToBGR = ::MNNNV21ToBGRRVV;
    gInstance->MNNSoftmax = ::MNNSoftmaxRVV;

    gInt8Instance->Int8GemmKernel = ::MNNGemmInt8AddBiasScale_16x4_Unit_RVV;
    gInt8Instance->Int8GemmKernel_W4 = ::MNNGemmInt8AddBiasScale_16x4_w4_Unit_RVV;

    gInstance->int8MatmulRelatedFunctions.Int8GemmKernel = gInt8Instance->Int8GemmKernel;
    gInstance->int8MatmulRelatedFunctions.Int8GemmKernel_W4 = gInt8Instance->Int8GemmKernel_W4;

    return true;
}

CoreFunctions* RVVFunctions::get() {
    return gInstance;
}

CoreInt8Functions* RVVFunctions::getInt8() {
    return gInt8Instance;
}

} // namespace MNN
