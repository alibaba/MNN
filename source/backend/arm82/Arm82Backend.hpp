//
//  Arm82Backend.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#ifndef Arm82Backend_hpp
#define Arm82Backend_hpp

#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include <MNN/HalideRuntime.h>

// armv82's data type default is fp16, so set
// armv82's dataformat: NC8HW8
#define ARMV82_CHANNEL_UNIT 8

typedef __fp16 FLOAT16;
template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<FLOAT16>() {
    return halide_type_t(halide_type_float, 16);
}

namespace MNN {
class Arm82Backend : public CPUBackend {
public:
    virtual ~Arm82Backend();
    Arm82Backend(const CPURuntime* runtime, BackendConfig::MemoryMode memory);
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
    virtual Backend::MemObj* onAcquire(const Tensor* nativeTensor, StorageType storageType) override;

    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

    int numberThread() const {
        return threadNumber();
    }
public:
    class Arm82Creator {
    public:
        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op, Backend* backend) const = 0;
    };

    static bool addArm82Creator(OpType t, Arm82Creator* ct);
};

#define REGISTER_ARM82_OP_CREATOR(type, creator) \
    void ___##type##__##creator##__() { \
        Arm82Backend::addArm82Creator(type, new creator); \
    }

inline int ARM82TensorElementSizeHelper(const Tensor* t) {
    int size = 1;
    for (int i = 0; i < t->dimensions(); i++) {
        int currentDimSize = t->length(i);
        if (TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && 1 == i) {
            currentDimSize = UP_DIV(currentDimSize, 8) * 8;
        }
        size *= currentDimSize;
    }
    return size;
}

inline int ARM82TensorStrideHelper(const Tensor* t, int dim) {
    int size = 1;
    for (int i = t->dimensions() - 1; i > dim; i--) {
        int currentDimSize = t->length(i);
        if (TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && 1 == i) {
            currentDimSize = UP_DIV(currentDimSize, 8) * 8;
        }
        size *= currentDimSize;
    }
    return size;
}

} // namespace MNN

#endif /* Arm82Backend_hpp */
#endif
