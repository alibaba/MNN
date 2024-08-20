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
#include "backend/cpu/CPUCast.hpp"
extern "C" {
void MNNInt8ToUInt8(void* ptr, int count);
void MNNUInt8ToInt8(void* ptr, int count);
}

namespace MNN {
bool AVX2Backend::isValid() {
    return nullptr != AVX2Functions::get();
}

AVX2Backend::AVX2Backend(const CPURuntime* runtime, BackendConfig::MemoryMode memory, size_t flags) : CPUBackend(runtime, BackendConfig::Precision_Low, memory, MNN_FORWARD_CPU_EXTENSION, flags) {
    mCoreFunctions = AVX2Functions::get();
    mInt8CoreFunctions = AVX2Functions::getInt8();
}

AVX2Backend::~AVX2Backend() {
    // nothing to do
}
// TODO: Move to functions

static void _CopyC16ToC4_int8(float* dstO, const float* srcO, int channelC4, int area) {
    auto dst = (int32_t*)dstO;
    auto src = (int32_t*)srcO;
    int c8 = channelC4 / 4;
    int cR = channelC4 % 4;
    for (int z=0; z<c8; ++z) {
        auto s0 = dst + 4 * z * area;
        auto s1 = dst + (4 * z + 1) * area;
        auto s2 = dst + (4 * z + 2) * area;
        auto s3 = dst + (4 * z + 3) * area;
        auto d = src + z * area * 4;
        for (int x=0; x<area; ++x) {
            *s0 = d[0];
            *s1 = d[1];
            *s2 = d[2];
            *s3 = d[3];
            s0++;
            s1++;
            s2++;
            s3++;
            d+=4;
        }
    }
    if (cR > 0) {
        auto s0 = dst + 4 * c8 * area;
        auto d = src + c8 * area * 4;
        for (int x=0; x<area; ++x) {
            for (int v=0; v<cR; ++v) {
                s0[v * area] = d[v];
            }
            s0++;
            d+=4;
        }
    }
}


static void _CopyC4ToC16_int8(float* dstO, const float* srcO, int channelC4, int area) {
    auto dst = (int32_t*)dstO;
    auto src = (int32_t*)srcO;
    int c8 = channelC4 / 4;
    int cR = channelC4 % 4;
    for (int z=0; z<c8; ++z) {
        auto s0 = src + 4 * z * area;
        auto s1 = src + (4 * z + 1) * area;
        auto s2 = src + (4 * z + 2) * area;
        auto s3 = src + (4 * z + 3) * area;
        auto d = dst + z * area * 4;
        for (int x=0; x<area; ++x) {
            d[0] = *s0;
            d[1] = *s1;
            d[2] = *s2;
            d[3] = *s3;
            s0 ++;
            s1 ++;
            s2 ++;
            s3 ++;
            d += 4;
        }
    }
    if (cR > 0) {
        auto s0 = src + 4 * c8 * area;
        auto d = dst + c8 * area * 4;
        for (int x=0; x<area; ++x) {
            for (int v=0; v<cR; ++v) {
                d[v] = s0[v * area];
            }
            for (int v=cR; v<4; ++v) {
                d[v] = 0;
            }
            s0 ++;
            d += 4;
        }
    }
}

static void _CopyC4ToC16(float* dst, const float* src, int channelC4, int area) {
    int c8 = channelC4 / 4;
    int cR = channelC4 % 4;
    for (int z=0; z<c8; ++z) {
        auto s0 = src + 4 * z * area * 4;
        auto s1 = src + (4 * z + 1) * area * 4;
        auto s2 = src + (4 * z + 2) * area * 4;
        auto s3 = src + (4 * z + 3) * area * 4;
        auto d = dst + z * area * 16;
        for (int x=0; x<area; ++x) {
            auto v0 = _mm_loadu_ps(s0);
            auto v1 = _mm_loadu_ps(s1);
            auto v2 = _mm_loadu_ps(s2);
            auto v3 = _mm_loadu_ps(s3);
            _mm_storeu_ps(d + 0, v0);
            _mm_storeu_ps(d + 4, v1);
            _mm_storeu_ps(d + 8, v2);
            _mm_storeu_ps(d + 12, v3);
            s0 += 4;
            s1 += 4;
            s2 += 4;
            s3 += 4;
            d += 16;
        }
    }
    if (cR > 0) {
        auto s0 = src + 4 * c8 * area * 4;
        auto d = dst + c8 * area * 16;
        auto v1 = _mm_setzero_ps();
        for (int x=0; x<area; ++x) {
            for (int v=0; v<cR; ++v) {
                auto v0 = _mm_loadu_ps(s0 + v * area * 4);
                _mm_storeu_ps(d + 4 * v, v0);
            }
            for (int v=cR; v<4; ++v) {
                _mm_storeu_ps(d + 4 * v, v1);
            }
            s0 += 4;
            d += 16;
        }
    }
}

static void _CopyC16ToC4(float* dst, const float* src, int channelC4, int area) {
    int c8 = channelC4 / 4;
    int cR = channelC4 % 4;
    for (int z=0; z<c8; ++z) {
        auto s0 = dst + 4 * z * area * 4;
        auto s1 = dst + (4 * z + 1) * area * 4;
        auto s2 = dst + (4 * z + 2) * area * 4;
        auto s3 = dst + (4 * z + 3) * area * 4;
        auto d = src + z * area * 16;
        for (int x=0; x<area; ++x) {
            auto v0 = _mm_loadu_ps(d);
            auto v1 = _mm_loadu_ps(d + 4);
            auto v2 = _mm_loadu_ps(d + 8);
            auto v3 = _mm_loadu_ps(d + 12);
            _mm_storeu_ps(s0, v0);
            _mm_storeu_ps(s1, v1);
            _mm_storeu_ps(s2, v2);
            _mm_storeu_ps(s3, v3);
            s0 += 4;
            s1 += 4;
            s2 += 4;
            s3 += 4;
            d+= 16;
        }
    }
    if (cR > 0) {
        auto s0 = dst + 4 * c8 * area * 4;
        auto d = src + c8 * area * 16;
        for (int x=0; x<area; ++x) {
            for (int v=0; v<cR; ++v) {
                auto v0 = _mm_loadu_ps(d + v * 4);
                _mm_storeu_ps(s0 + 4 * v * area, v0);
            }
            s0 += 4;
            d+= 16;
        }
    }
}

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

static void _CopyC4ToC8_int8(float* dstPtr, const float* srcPtr, int channelC4, int area) {
    int8_t* dst = (int8_t*)(dstPtr);
    const int8_t* src = (const int8_t*)(srcPtr);
    int c8 = channelC4 / 2;
    int cR = channelC4 % 2;
    for (int z=0; z<c8; ++z) {
        auto s0 = src + 2 * z * area * 4;
        auto s1 = src + (2 * z + 1) * area * 4;
        auto d = dst + z * area * 8;
        for (int x=0; x<area; ++x) {
            *(int*)d = *(int*)s0;
            *((int*)d + 1) = *(int*)s1;
            s0 += 4;
            s1 += 4;
            d += 8;
        }
    }
    if (cR > 0) {
        auto s0 = src + 2 * c8 * area * 4;
        auto d = dst + c8 * area * 8;
        for (int x=0; x<area; ++x) {
            *(int*)d = *(int*)s0;
            *((int*)d + 1) = 0;
            s0 += 4;
            d += 8;
        }
    }
}

static void _CopyC8ToC4_int8(float* dstPtr, const float* srcPtr, int channelC4, int area) {
    int8_t* dst = (int8_t*)(dstPtr);
    const int8_t* src = (const int8_t*)(srcPtr);
    int c8 = channelC4 / 2;
    int cR = channelC4 % 2;
    for (int z=0; z<c8; ++z) {
        auto s0 = dst + 2 * z * area * 4;
        auto s1 = dst + (2 * z + 1) * area * 4;
        auto d = src + z * area * 8;
        for (int x=0; x<area; ++x) {
            *(int*)s0 = *(int*)d;
            *(int*)s1 = *((int*)d + 1);
            s0 += 4;
            s1 += 4;
            d+= 8;
        }
    }
    if (cR > 0) {
        auto s0 = dst + 2 * c8 * area * 4;
        auto d = src + c8 * area * 8;
        for (int x=0; x<area; ++x) {
            *(int*)s0 = *(int*)d;
            s0 += 4;
            d += 8;
        }
    }
}

Execution* AVX2Backend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const MNN::Op* op) {
    if (op->type() == OpType_ImageProcess) {
        return CPUBackend::onCreate(inputs, outputs, op);
    }
    for (auto t : outputs) {
        if (t->getType().code != halide_type_float && t->getType().bits != 8) {
            return nullptr;
        }
        if (t->getType().code == halide_type_uint) {
            return nullptr;
        }
    }
    bool originCreate = OpCommonUtils::opCompabilityForLowp(op, 4);
    if (originCreate || op->type() == OpType_Softmax || op->type() == OpType_Reduction || op->type() == OpType_ConvInt8 || op->type() == OpType_DepthwiseConvInt8 || op->type() == OpType_FloatToInt8 || op->type() == OpType_Int8ToFloat) {
        return CPUBackend::onCreate(inputs, outputs, op);
    }
    return nullptr;
}

Backend::MemObj* AVX2Backend::onAcquire(const Tensor* nativeTensor, StorageType storageType) {
    // arm82 backend tensor data type is fp16 default
    auto tensor = const_cast<Tensor*>(nativeTensor);
    auto& buffer = tensor->buffer();
    auto tensorSize = getTensorSize(nativeTensor, true);
    // MNN_PRINT("acquire tensor:%p, tensorSize:%d, shape: ", nativeTensor, tensorSize);
    // nativeTensor->printShape();
    auto res = allocBuffer(tensorSize, (Tensor*)nativeTensor, storageType);
    if (!res) {
        return nullptr;
    }
    // Set mask in device for easy to determine
    buffer.device = 1;
    return res;
}

void AVX2Backend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto& ib     = srcTensor->buffer();
    auto& ob     = dstTensor->buffer();
    std::unique_ptr<Tensor> wrapTensor;
    if (ib.type.code != halide_type_float && ib.type != halide_type_of<int8_t>()) {
        CPUBackend::onCopyBuffer(srcTensor, dstTensor);
        return;
    }
    if (ib.dimensions <= 1) {
        CPUBackend::onCopyBuffer(srcTensor, dstTensor);
        return;
    }
    if (getDataType(srcTensor) != getDataType(dstTensor)) {
        auto dimType = Tensor::CAFFE;
        switch (TensorUtils::getDescribe(srcTensor)->dimensionFormat) {
            case MNN_DATA_FORMAT_NCHW:
                break;
            case MNN_DATA_FORMAT_NC4HW4:
                dimType = Tensor::CAFFE_C4;
                break;
            case MNN_DATA_FORMAT_NHWC:
                dimType = Tensor::TENSORFLOW;
                break;
            default:
                break;
        }
        auto convertType = CPUCastCreator::FlOAT_TO_INT8;
        if (getDataType(srcTensor) == DataType_DT_INT8) {
            convertType = CPUCastCreator::INT8_TO_FlOAT;
        }
        wrapTensor.reset(Tensor::createDevice(srcTensor->shape(), dstTensor->getType(), dimType));
        auto dstType = getDataType(dstTensor);
        if (dstType != DataType_DT_FLOAT) {
            wrapTensor->setType(dstType);
        }
        wrapTensor->buffer().host = (uint8_t*)MNNMemoryAllocAlign(getTensorSize(wrapTensor.get()) * wrapTensor->getType().bytes(), MNN_MEMORY_ALIGN_DEFAULT);
        TensorUtils::getDescribe(wrapTensor.get())->memoryType = Tensor::InsideDescribe::MEMORY_HOST;
        auto code = CPUCastCreator::cast(srcTensor, wrapTensor.get(), this, convertType);
        if (NO_ERROR != code) {
            MNN_ERROR("Error in CPUBackend::onCopyBuffer:cast\n");
        }
        srcTensor = wrapTensor.get();
    } else if (srcTensor->getType() != dstTensor->getType()) {
       MNN_ERROR("Input type not match session's tensor\n");
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
        auto outF = _CopyC8ToC4;
        auto inF = _CopyC4ToC8;
        auto obBytes = CPUBackend::getBytes(this, dstTensor);
        if (obBytes == 1) {
            outF = _CopyC8ToC4_int8;
            inF = _CopyC4ToC8_int8;
        }
        if (mCoreFunctions->pack == 16) {
            outF = _CopyC16ToC4;
            inF = _CopyC4ToC16;
            if (obBytes == 1) {
                outF = _CopyC16ToC4_int8;
                inF = _CopyC4ToC16_int8;
            }
        }
        // NC4HW4 <-> NC8HW8
        if (1 == srcTensor->dimensions()) {
            ::memcpy(dstTensor->host<void>(), srcTensor->host<void>(), srcTensor->length(0) * srcTensor->getType().bytes());
            return;
        }
        auto dims = CPUTensorConverter::splitDimensions(srcTensor->buffer(), source);
        int area = std::get<1>(dims) * std::get<0>(dims);
        int channel = std::get<2>(dims);
        auto c4 = UP_DIV(channel, 4);
        if (srcType == MNN_FORWARD_CPU_EXTENSION) {
            outF(dstTensor->host<float>(), srcTensor->host<float>(), c4, area);
        } else {
            inF(dstTensor->host<float>(), srcTensor->host<float>(), c4, area);
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
