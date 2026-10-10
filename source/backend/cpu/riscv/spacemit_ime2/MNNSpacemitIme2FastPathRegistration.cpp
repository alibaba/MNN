//
//  MNNSpacemitIme2FastPathRegistration.cpp
//  MNN
//
//  Created by MNN on 2026/07/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/CPUExtension.hpp"

#ifdef MNN_LOW_MEMORY
#include "MNNSpacemitIme2ConvInt8Executor.hpp"
#endif
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include "MNNSpacemitIme2Attention.hpp"
#endif

extern void MNNSpacemitIme2AttentionSoftmax(float* softmaxDst, const float* softmaxSrc, float* runningMax,
                                            float* runningSum, float* updateScale, int outside, int reduceSize,
                                            int kvSeqOffset, int validOffset, int pack, bool mask);
extern void MNNSpacemitIme2GemmInt8AddBiasScaleW8(int8_t* dst, const int8_t* src, const int8_t* weight,
                                                  size_t srcDepthQuad, size_t dstStep, size_t dstDepthQuad,
                                                  const QuanPostTreatParameters* post, size_t realCount);
#ifdef MNN_LOW_MEMORY
extern void MNNSpacemitIme2GemmInt8AddBiasScaleW4(int8_t* dst, const int8_t* src, const int8_t* weight,
                                                  size_t srcDepthQuad, size_t dstStep, size_t dstDepthQuad,
                                                  const QuanPostTreatParameters* post, size_t realCount);
#endif

namespace MNN {

void MNNRvvInitializeFastPathFunctions(CoreFunctions* core) {
    core->MNNSoftmax = MNNSpacemitIme2AttentionSoftmax;
#ifdef MNN_LOW_MEMORY
    constexpr CPUExtension::CreateInt8GemmExecution createInt8Gemm = MNNSpacemitIme2CreateInt8GemmExecution;
#else
    constexpr CPUExtension::CreateInt8GemmExecution createInt8Gemm = nullptr;
#endif
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
    static constexpr CPUExtension extension(createInt8Gemm, MNNSpacemitIme2CreateAttentionExecution);
#else
    static constexpr CPUExtension extension(createInt8Gemm, nullptr);
#endif
    core->kvUpdateConcurrent = true;
    core->extension = &extension;
}

void MNNRvvInitializeInt8FastPathFunctions(CoreInt8Functions* core) {
    core->Int8GemmKernel = MNNSpacemitIme2GemmInt8AddBiasScaleW8;
#ifdef MNN_LOW_MEMORY
    core->Int8GemmKernel_W4 = MNNSpacemitIme2GemmInt8AddBiasScaleW4;
#endif
}

} // namespace MNN
