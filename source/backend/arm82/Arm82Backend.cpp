//
//  Arm82Backend.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <mutex>

#include "backend/arm82/Arm82Backend.hpp"
#include "backend/arm82/Arm82OptFunc.hpp"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"

#include "half.hpp"

namespace MNN {

static const MNNForwardType gForwardType = MNN_FORWARD_CPU_EXTENSION;

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

Arm82Backend::Arm82Backend(Backend* cpuBackend) : Backend(gForwardType), mCPUBackend(cpuBackend) {
    // nonthing to do
}

Arm82Backend::~Arm82Backend() {
}

Execution* Arm82Backend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const MNN::Op* op) {
    auto creatorContainer = getArm82CreatorContainer();
    // MNN_PRINT("====> create Execution for type: %s\n", MNN::EnumNameOpType(op->type()));
    auto iter = creatorContainer->find(op->type());

    if (op->type() == OpType_BinaryOp) {
        auto param      = op->main_as_BinaryOp();
        auto binaryType = param->opType();
        if (binaryType == BinaryOpOperation_ADD) {
            std::shared_ptr<OpT> opTemp(op->UnPack());

            opTemp->type                   = OpType_Eltwise;
            opTemp->main.type              = OpParameter_Eltwise;
            opTemp->main.value             = new EltwiseT;
            opTemp->main.AsEltwise()->type = EltwiseType_SUM;

            flatbuffers::FlatBufferBuilder builder;
            auto offset = Op::Pack(builder, opTemp.get());
            builder.Finish(offset);
            auto eleOp = flatbuffers::GetMutableRoot<Op>(builder.GetBufferPointer());

            auto iter = creatorContainer->find(OpType_Eltwise);
            auto exe  = iter->second->onCreate(inputs, outputs, eleOp, this);
            return exe;
        }
    }

    if (iter == creatorContainer->end()) {
        //MNN_PRINT("[MNNWarning]: ARMV82 don't support type: [%s], %s\n", MNN::EnumNameOpType(op->type()),
        //          op->name()->c_str());
        return nullptr;
    }
    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (exe == nullptr) {
        //MNN_PRINT("[MNNWarning]: ARMV82 don't support type: [%s], %s\n", MNN::EnumNameOpType(op->type()),
        //          op->name()->c_str());
        return nullptr;
    }
    return exe;
}

bool Arm82Backend::onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) {
    // arm82 backend tensor data type is fp16 default
    auto tensor = const_cast<Tensor*>(nativeTensor);

    auto& tensorBuffer = tensor->buffer();
    auto size          = tensorBuffer.type.bytes();
    // The default data type of input tensor for arm82 backend is FLOAT32.
    // However, Arm82Backend default data type is FLOAT16, so check whether data type is FLOAT32,
    // then divide size by 2
    if (tensorBuffer.type == halide_type_of<float>()) {
        size /= 2;
    }

    const int dimensions = tensorBuffer.dimensions;
    for (int i = 0; i < dimensions; i++) {
        int currentDimSize = tensorBuffer.dim[i].extent;
        if (TensorUtils::getDescribe(tensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && 1 == i) {
            currentDimSize = ALIGN_UP8(currentDimSize);
        }
        size *= currentDimSize;
    }

    if (size <= 0) {
        MNN_ERROR("[MNN ERROR]tensor size is less than zero!\n");
        return false;
    }

    auto cpuBufferAllocator = static_cast<BufferAllocator*>(mCPUBackend->getAllocator(storageType));
    switch (storageType) {
        case Backend::STATIC:
        case Backend::DYNAMIC:
            tensorBuffer.host = (uint8_t*)cpuBufferAllocator->alloc(size, false);
            break;
        case Backend::DYNAMIC_SEPERATE:
            tensorBuffer.host = (uint8_t*)cpuBufferAllocator->alloc(size, true);
            break;
        default:
            tensorBuffer.host = (uint8_t*)cpuBufferAllocator->alloc(size, false);
            break;
    }

    if (nullptr == tensorBuffer.host) {
        MNN_ERROR("Alloc buffer ERROR for Arm82Backend\n");
        return false;
    }

    return true;
}
bool Arm82Backend::onReleaseBuffer(const Tensor* nativeTensor, StorageType storageType) {
    return mCPUBackend->onReleaseBuffer(nativeTensor, storageType);
}
bool Arm82Backend::onAllocateBuffer() {
    return mCPUBackend->onAllocateBuffer();
}
bool Arm82Backend::onClearBuffer() {
    return mCPUBackend->onClearBuffer();
}

void Arm82Backend::onExecuteBegin() const {
    return mCPUBackend->onExecuteBegin();
}
void Arm82Backend::onExecuteEnd() const {
    return mCPUBackend->onExecuteEnd();
}

void Arm82Backend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto ib     = srcTensor->buffer();
    auto ob     = dstTensor->buffer();
    auto source = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    auto dest   = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
    auto srcBn  = TensorUtils::getDescribe(srcTensor)->backend;
    auto dstBn  = TensorUtils::getDescribe(dstTensor)->backend;

    auto fastMode = source == dest && (source == MNN_DATA_FORMAT_NCHW || source == MNN_DATA_FORMAT_NHWC);
    if (ib.dimensions <= 1 || fastMode) {
        const int elemenSize = srcTensor->elementSize();
        // cpu -> arm82 copy
        if (srcBn == mCPUBackend || dstBn == this) {
            const auto src = srcTensor->host<float>();
            auto dst       = dstTensor->host<FLOAT16>();
            MNNQuantizeFP16(dst, src, elemenSize);
            return;
        }
        // arm82 -> cpu copy
        if (srcBn == this || dstBn == mCPUBackend) {
            const auto src = srcTensor->host<half_float::half>();
            auto dst       = dstTensor->host<float>();
            for (int i = 0; i < elemenSize; ++i) {
                dst[i] = float(src[i]);
            }
            return;
        }
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

        for (int i = 0; i < batch; ++i) {
            MNNNC8HW8TONCHW((float*)ob.host + outBatchStide * i, (const uint16_t*)ib.host + inbatchStride * i, area,
                            channel);
        }
        return;
    }

    if (source == MNN_DATA_FORMAT_NCHW && dest == MNN_DATA_FORMAT_NC4HW4) {
        const int inbatchStride = channel * area;
        const int outBatchStide = UP_DIV(channel, ARMV82_CHANNEL_UNIT) * area * ARMV82_CHANNEL_UNIT;
        for (int i = 0; i < batch; ++i) {
            MNNNCHWTONC8HW8((uint16_t*)ob.host + outBatchStide * i, (const float*)ib.host + inbatchStride * i, area,
                            channel);
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
    if (srcBn == this || dstBn == mCPUBackend) {
        const int inbatchStride = ROUND_UP(channel, ARMV82_CHANNEL_UNIT) * area;
        const int outBatchStide = ob.dim[0].stride;
        for (int i = 0; i < batch; ++i) {
            MNNNC8HW8TONC4HW4((float*)ob.host + outBatchStide * i, (const uint16_t*)ib.host + inbatchStride * i, area,
                              channel);
        }
        return;
    }

    if (srcBn == mCPUBackend || dstBn == this) {
        const int inbatchStride = ib.dim[0].stride;
        const int outBatchStide = ROUND_UP(channel, ARMV82_CHANNEL_UNIT) * area;
        for (int i = 0; i < batch; ++i) {
            MNNNC4HW4TONC8HW8((uint16_t*)ob.host + outBatchStide * i, (const float*)ib.host + inbatchStride * i, area,
                              channel);
        }
        return;
    }

    MNN_ERROR("[MNN ERROR] Arm82Backend do not support convert from [%s] to [%s]\n", EnumNameMNN_DATA_FORMAT(source),
              EnumNameMNN_DATA_FORMAT(dest));
    return;
}

int Arm82Backend::numberThread() const {
    return static_cast<CPUBackend*>(mCPUBackend)->threadNumber();
}

class Arm82BackendCreator : public BackendCreator {
public:
    virtual Backend* onCreate(const Backend::Info& info) const override {
        if (info.user == nullptr || info.user->sharedContext == nullptr) {
            return nullptr;
        }
        return new Arm82Backend(static_cast<CPUBackend*>(info.user->sharedContext));
    };
};

#if defined(__aarch64__) && defined(__APPLE__)
void registerArm82BackendCreator() {
    MNNInsertExtraBackendCreator(gForwardType, new Arm82BackendCreator);
};

#else

static bool gResistor = []() {
    MNNInsertExtraBackendCreator(gForwardType, new Arm82BackendCreator);
    return true;
}();

#endif

} // namespace MNN
