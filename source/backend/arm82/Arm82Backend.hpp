//
//  Arm82Backend.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Arm82Backend_hpp
#define Arm82Backend_hpp

#include "MNN_generated.h"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Backend.hpp"
#include "core/Macro.h"

// armv82's data type default is fp16, so set
// armv82's dataformat: NC8HW8
#define ARMV82_CHANNEL_UNIT 8

typedef __fp16 FLOAT16;

namespace MNN {
class Arm82Backend : public Backend {
public:
    virtual ~Arm82Backend();
    Arm82Backend(Backend* cpuBackend);
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
    virtual bool onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) override;
    virtual bool onReleaseBuffer(const Tensor* nativeTensor, StorageType storageType) override;
    virtual bool onAllocateBuffer() override;
    virtual bool onClearBuffer() override;

    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;

    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

    int numberThread() const;
#ifdef MNN_USE_THREAD_POOL
    inline int taskIndex() const {
        return static_cast<CPUBackend*>(mCPUBackend)->taskIndex();
    }
#endif
public:
    class Arm82Creator {
    public:
        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op, Backend* backend) const = 0;
    };

    static bool addArm82Creator(OpType t, Arm82Creator* ct);

private:
    Backend* mCPUBackend;
};

#define REGISTER_ARM82_OP_CREATOR(type, creator)          \
    static bool gRegister##type = []() {                  \
        Arm82Backend::addArm82Creator(type, new creator); \
        return true;                                      \
    }();

template <typename T, int UNIT>
void MyPrint(const T* data, int size) {
    for (int i = 0; i < size; ++i) {
        if (i % UNIT == 0) {
            MNN_PRINT("\n%d: ", i / UNIT);
        }
        MNN_PRINT("%f, ", float(data[i]));
    }
    MNN_PRINT("\n");
}

// use for computing stride for nc4hw4(a.k.a nc8hw8)
inline int ARM82TensorBatchStrideHelper(const Tensor* t) {
    int channel = t->channel();
    return t->height() * t->width() * ROUND_UP(channel, ARMV82_CHANNEL_UNIT);
}

} // namespace MNN

#endif /* Arm82Backend_hpp */
