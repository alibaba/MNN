//
//  AVX2Backend.cpp
//  MNN
//
//  Created by MNN on 2021/05/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include "AVX2Functions.hpp"
#include "AVX2Backend.hpp"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "backend/cpu/CPURaster.hpp"
#include "backend/cpu/CPUReduction.hpp"
#include "backend/cpu/CPUSoftmax.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {
bool AVX2Backend::isValid() {
    return nullptr != AVX2Functions::get();
}

AVX2Backend::AVX2Backend(const CPURuntime* runtime, size_t flags) : CPUBackend(runtime, BackendConfig::Precision_Low, MNN_FORWARD_CPU_EXTENSION, flags) {
    mCoreFunctions = AVX2Functions::get();
}

AVX2Backend::~AVX2Backend() {
    // nothing to do
}

// TODO: Move to functions
static void _CopyC4ToC8(float* dst, const float* src, int channelC4, int area) {
    int c8 = channelC4 / 2;
    int cR = channelC4 % 2;
    for (int z=0; z<c8; ++z) {
        auto s0 = src + 2 * z * area * 4;
        auto s1 = src + (2 * z + 1) * area * 4;
        auto d = dst + z * area * 8;
        for (int x=0; x<area; ++x) {
            auto v0 = _mm_loadu_ps(s0);
            auto v1 = _mm_loadu_ps(s1);
            _mm_storeu_ps(d + 0, v0);
            _mm_storeu_ps(d + 4, v1);
            s0 += 4;
            s1 += 4;
            d += 8;
        }
    }
    if (cR > 0) {
        auto s0 = src + 2 * c8 * area * 4;
        auto d = dst + c8 * area * 8;
        auto v1 = _mm_setzero_ps();
        for (int x=0; x<area; ++x) {
            auto v0 = _mm_loadu_ps(s0);
            _mm_storeu_ps(d + 0, v0);
            _mm_storeu_ps(d + 4, v1);
            s0 += 4;
            d += 8;
        }
    }
}

static void _CopyC8ToC4(float* dst, const float* src, int channelC4, int area) {
    int c8 = channelC4 / 2;
    int cR = channelC4 % 2;
    for (int z=0; z<c8; ++z) {
        auto s0 = dst + 2 * z * area * 4;
        auto s1 = dst + (2 * z + 1) * area * 4;
        auto d = src + z * area * 8;
        for (int x=0; x<area; ++x) {
            auto v0 = _mm_loadu_ps(d);
            auto v1 = _mm_loadu_ps(d + 4);
            _mm_storeu_ps(s0, v0);
            _mm_storeu_ps(s1, v1);
            s0 += 4;
            s1 += 4;
            d+= 8;
        }
    }
    if (cR > 0) {
        auto s0 = dst + 2 * c8 * area * 4;
        auto d = src + c8 * area * 8;
        for (int x=0; x<area; ++x) {
            auto v0 = _mm_loadu_ps(d);
            _mm_storeu_ps(s0, v0);
            s0 += 4;
            d+= 8;
        }
    }
}

Execution* AVX2Backend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const MNN::Op* op) {
    for (auto t : outputs) {
        if (t->getType().code != halide_type_float) {
            return nullptr;
        }
    }
    auto quantInfo = OpCommonUtils::getQuantInfo(inputs);
    if (quantInfo.first) {
        return nullptr;
    }
    bool originCreate = OpCommonUtils::opCompabilityForLowp(op);
    if (originCreate) {
        return CPUBackend::onCreate(inputs, outputs, op);
    }
    if (op->type() == OpType_Raster) {
        return new CPURaster(this);
    }
    if (op->type() == OpType_Reduction) {
        return makePostWrapExectuion(CPUReductionCreator::create(inputs, outputs, op, this));
    }
    if (op->type() == OpType_Softmax) {
        return makePostWrapExectuion(CPUSoftmax::create(op, this));
    }
    return nullptr;
}

bool AVX2Backend::onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) {
    // arm82 backend tensor data type is fp16 default
    auto tensor = const_cast<Tensor*>(nativeTensor);
    auto& buffer = tensor->buffer();
    if (buffer.type != halide_type_of<float>()) {
        return CPUBackend::onAcquireBuffer(nativeTensor, storageType);
    }
    auto tensorSize = getTensorSize(nativeTensor);
    auto res = allocBuffer(tensorSize * sizeof(float), (Tensor*)nativeTensor, storageType);
    if (!res) {
        return false;
    }
    // Set mask in device for easy to determine
    buffer.device = 1;
    return true;
}

void AVX2Backend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto& ib     = srcTensor->buffer();
    auto& ob     = dstTensor->buffer();
    if (ib.type.code != halide_type_float) {
        CPUBackend::onCopyBuffer(srcTensor, dstTensor);
        return;
    }
    auto source = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    auto dest   = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
    auto srcType = MNN_FORWARD_CPU;
    if (ib.device != 0) {
        srcType = MNN_FORWARD_CPU_EXTENSION;
    }
    auto dstType = MNN_FORWARD_CPU;
    if (ob.device != 0) {
        dstType = MNN_FORWARD_CPU_EXTENSION;
    }
    if (srcType == dstType) {
        if(srcType == MNN_FORWARD_CPU_EXTENSION) {
            CPUTensorConverter::convert(srcTensor, dstTensor, mCoreFunctions);
        } else {
            CPUTensorConverter::convert(srcTensor, dstTensor, MNNGetCoreFunctions());
        }
        return;
    }
    if (source != MNN_DATA_FORMAT_NC4HW4 && dest != MNN_DATA_FORMAT_NC4HW4) {
        CPUTensorConverter::convert(srcTensor, dstTensor, mCoreFunctions);
        return;
    }
    if (source == MNN_DATA_FORMAT_NC4HW4 && dest == MNN_DATA_FORMAT_NC4HW4) {
        // NC4HW4 <-> NC8HW8
        if (1 == srcTensor->dimensions()) {
            ::memcpy(dstTensor->host<void>(), srcTensor->host<void>(), srcTensor->length(0) * srcTensor->getType().bytes());
            return;
        }
        auto dims = CPUTensorConverter::splitDimensions(srcTensor->buffer(), source);
        int batch = std::get<0>(dims);
        int area = std::get<1>(dims);
        int channel = std::get<2>(dims);
        auto c4 = UP_DIV(channel, 4);
        auto c8 = UP_DIV(channel, mCoreFunctions->pack);
        auto c4Stride = area * c4 * 4;
        auto c8Stride = area * c8 * mCoreFunctions->pack;
        if (srcType == MNN_FORWARD_CPU_EXTENSION) {
            for (int i=0; i<batch; ++i) {
                _CopyC8ToC4(dstTensor->host<float>() + i*c4Stride, srcTensor->host<float>() + i * c8Stride, c4, area);
            }
        } else {
            for (int i=0; i<batch; ++i) {
                _CopyC4ToC8(dstTensor->host<float>() + i*c8Stride, srcTensor->host<float>() + i * c4Stride, c4, area);
            }
        }
        return;
    }
    if (source == MNN_DATA_FORMAT_NC4HW4) {
        if (srcType == MNN_FORWARD_CPU_EXTENSION) {
            CPUTensorConverter::convert(srcTensor, dstTensor, mCoreFunctions);
        } else {
            CPUTensorConverter::convert(srcTensor, dstTensor, MNNGetCoreFunctions());
        }
        return;
    }
    if (dest == MNN_DATA_FORMAT_NC4HW4) {
        if (dstType == MNN_FORWARD_CPU_EXTENSION) {
            CPUTensorConverter::convert(srcTensor, dstTensor, mCoreFunctions);
        } else {
            CPUTensorConverter::convert(srcTensor, dstTensor, MNNGetCoreFunctions());
        }
        return;
    }
    MNN_ASSERT(false);
    return;
}

} // namespace MNN
