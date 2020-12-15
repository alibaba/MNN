//
//  Arm82Backend.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include <algorithm>
#include <mutex>

#include "backend/arm82/Arm82Backend.hpp"
#include "backend/arm82/Arm82OptFunc.hpp"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"

#include "half.hpp"

namespace MNN {

void registerArm82Ops();

static inline std::map<OpType, Arm82Backend::Arm82Creator*>* getArm82CreatorContainer() {
    static std::once_flag fg;
    static std::map<OpType, Arm82Backend::Arm82Creator*>* ret = nullptr;
    std::call_once(fg, [&] { ret = new std::map<OpType, Arm82Backend::Arm82Creator*>; });
    return ret;
}

bool Arm82Backend::addArm82Creator(OpType t, Arm82Creator* ct) {
    auto creatorContainer = getArm82CreatorContainer();
    if (creatorContainer->find(t) == creatorContainer->end()) {
        creatorContainer->insert(std::make_pair(t, ct));
    }
    return true;
}

Arm82Backend::Arm82Backend(const CPURuntime* runtime) : CPUBackend(runtime, MNN_FORWARD_CPU_EXTENSION) {
    // nothing to do
}

Arm82Backend::~Arm82Backend() {
    // nothing to do
}

Execution* Arm82Backend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const MNN::Op* op) {
    for (auto t : outputs) {
        if (t->getType().code != halide_type_float) {
            return nullptr;
        }
    }
    auto creatorContainer = getArm82CreatorContainer();
    // MNN_PRINT("====> create Execution for type: %s\n", MNN::EnumNameOpType(op->type()));
    auto iter = creatorContainer->find(op->type());

    if (iter == creatorContainer->end()) {
//        MNN_PRINT("[MNNWarning]: ARMV82 don't support type: [%s]\n", MNN::EnumNameOpType(op->type()));
        return nullptr;
    }
    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (exe == nullptr) {
//        MNN_PRINT("[MNNWarning]: ARMV82 don't support type: [%s]\n", MNN::EnumNameOpType(op->type()));
        return nullptr;
    }
    return exe;
}

bool Arm82Backend::onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) {
    // arm82 backend tensor data type is fp16 default
    auto tensor = const_cast<Tensor*>(nativeTensor);
    auto& buffer = tensor->buffer();
    if (buffer.type != halide_type_of<float>()) {
        return CPUBackend::onAcquireBuffer(nativeTensor, storageType);
    }
    // The default data type of input tensor for arm82 backend is FLOAT32.
    // However, Arm82Backend default data type is FLOAT16, so check whether data type is FLOAT32,
    // then divide size by 2
    int size          = sizeof(int16_t);
    const int dimensions = buffer.dimensions;
    for (int i = 0; i < dimensions; i++) {
        int currentDimSize = buffer.dim[i].extent;
        if (TensorUtils::getDescribe(tensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && 1 == i) {
            currentDimSize = ALIGN_UP8(currentDimSize);
        }
        size *= currentDimSize;
    }
    auto res = allocBuffer(size, (Tensor*)nativeTensor, storageType);
    if (!res) {
        return false;
    }
    // Set mask in device for easy to determine
    buffer.device = 1;
    return true;
}

void Arm82Backend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto ib     = srcTensor->buffer();
    auto ob     = dstTensor->buffer();
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
    //MNN_PRINT("%d, %d - %d, %d\n", source, srcType, dest, dstType);
    auto fastMode = source == dest && (source == MNN_DATA_FORMAT_NCHW || source == MNN_DATA_FORMAT_NHWC);
    //MNN_PRINT("%d -> %d, %d\n", source, dest, fastMode);
    if (ib.dimensions <= 1 || fastMode) {
        const int elemenSize = srcTensor->elementSize();
        // if not float, just copy data
        if(ib.type != halide_type_of<float>()){
            memcpy(dstTensor->host<char>(), srcTensor->host<char>(), srcTensor->size());
            return;
        }
        // copy and quantize/dequantize data
        // cpu -> arm82 copy
        if (srcType == MNN_FORWARD_CPU || dstType == MNN_FORWARD_CPU_EXTENSION) {
            const auto src = srcTensor->host<float>();
            auto dst       = dstTensor->host<FLOAT16>();
            MNNQuantizeFP16(dst, src, elemenSize);
            return;
        }
        // arm82 -> cpu copy
        if (srcType == MNN_FORWARD_CPU_EXTENSION || dstType == MNN_FORWARD_CPU) {
            const auto src = srcTensor->host<half_float::half>();
            auto dst       = dstTensor->host<float>();
            for (int i = 0; i < elemenSize; ++i) {
                dst[i] = float(src[i]);
            }
            return;
        }
        MNN_ASSERT(false);
    }

    int area    = 1;
    int channel = 0;
    if (source == MNN_DATA_FORMAT_NC4HW4 || source == MNN_DATA_FORMAT_NCHW) {
        channel = ib.dim[1].extent;
        for (int axis = 2; axis < ib.dimensions; ++axis) {
            area *= ib.dim[axis].extent;
        }
    } else {
        channel = ib.dim[ib.dimensions - 1].extent;
        for (int axis = 1; axis < ib.dimensions - 1; ++axis) {
            area *= ib.dim[axis].extent;
        }
    }

    // external use
    // copy between user and Arm82Backend
    // fp16 fp32 transformation
    const int batch = ib.dim[0].extent;

    if (source == MNN_DATA_FORMAT_NC4HW4 && dest == MNN_DATA_FORMAT_NCHW) {
        const int inbatchStride = UP_DIV(channel, ARMV82_CHANNEL_UNIT) * area * ARMV82_CHANNEL_UNIT;
        const int outBatchStide = channel * area;

        if(srcType == MNN_FORWARD_CPU_EXTENSION && dstType == MNN_FORWARD_CPU_EXTENSION){
            for (int i = 0; i < batch; ++i) {
                MNNNC8HW8TONCHW_NO_TYPE((uint16_t*)ob.host + outBatchStide * i, (const uint16_t*)ib.host + inbatchStride * i, area,
                                channel);
            }
        }else{
            for (int i = 0; i < batch; ++i) {
                MNNNC8HW8TONCHW((float*)ob.host + outBatchStide * i, (const uint16_t*)ib.host + inbatchStride * i, area,
                                channel);
            }
        }
        return;
    }

    if (source == MNN_DATA_FORMAT_NCHW && dest == MNN_DATA_FORMAT_NC4HW4) {
        const int inbatchStride = channel * area;
        const int outBatchStide = UP_DIV(channel, ARMV82_CHANNEL_UNIT) * area * ARMV82_CHANNEL_UNIT;
        if(dstType == MNN_FORWARD_CPU_EXTENSION && srcType == MNN_FORWARD_CPU_EXTENSION){
            for (int i = 0; i < batch; ++i) {
                MNNNCHWTONC8HW8_NO_TYPE((uint16_t*)ob.host + outBatchStide * i, (const uint16_t*)ib.host + inbatchStride * i, area,
                                channel);
            }
        }else{
            for (int i = 0; i < batch; ++i) {
                MNNNCHWTONC8HW8((uint16_t*)ob.host + outBatchStide * i, (const float*)ib.host + inbatchStride * i, area,
                                channel);
            }
        }
        return;
    }

    if (source == MNN_DATA_FORMAT_NC4HW4 && dest == MNN_DATA_FORMAT_NHWC) {
        const int inbatchStride = UP_DIV(channel, ARMV82_CHANNEL_UNIT) * area * ARMV82_CHANNEL_UNIT;
        const int outBatchStide = channel * area;

        for (int i = 0; i < batch; ++i) {
            MNNNC8HW8TONHWC((float*)ob.host + outBatchStide * i, (const uint16_t*)ib.host + inbatchStride * i, area,
                            channel);
        }
        return;
    }

    // internal use
    // copy between CPUBackend and Arm82Backend
    // Arm82Backend -> CPUBackend(Arm82Backend has not supported op, callback to CPUBackend)
    MNN_ASSERT(source == dest && source == MNN_DATA_FORMAT_NC4HW4);
    if (srcType == MNN_FORWARD_CPU_EXTENSION || dstType == MNN_FORWARD_CPU) {
        const int inbatchStride = ROUND_UP(channel, ARMV82_CHANNEL_UNIT) * area;
        const int outBatchStide = ob.dim[0].stride;
        for (int i = 0; i < batch; ++i) {
            MNNNC8HW8TONC4HW4((float*)ob.host + outBatchStide * i, (const uint16_t*)ib.host + inbatchStride * i, area,
                              channel);
        }
        return;
    }

    if (srcType == MNN_FORWARD_CPU || dstType == MNN_FORWARD_CPU_EXTENSION) {
        const int inbatchStride = ib.dim[0].stride;
        const int outBatchStide = ROUND_UP(channel, ARMV82_CHANNEL_UNIT) * area;
        for (int i = 0; i < batch; ++i) {
            MNNNC4HW4TONC8HW8((uint16_t*)ob.host + outBatchStide * i, (const float*)ib.host + inbatchStride * i, area,
                              channel);
        }
        return;
    }
    return;
}

void registerArm82RuntimeCreator() {
    registerArm82Ops();
};
#ifndef MNN_CODEGEN_REGISTER
static const auto __arm82_global_initializer = []() {
    registerArm82RuntimeCreator();
    return true;
}();
#endif

} // namespace MNN

#endif
