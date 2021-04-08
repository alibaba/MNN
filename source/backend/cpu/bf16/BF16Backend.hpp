//
//  BF16Backend.hpp
//  MNN
//
//  Created by MNN on 2020/01/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef BF16Backend_hpp
#define BF16Backend_hpp

#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
class BF16Backend : public CPUBackend {
public:
    virtual ~BF16Backend();
    BF16Backend(const CPURuntime* runtime);
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
    virtual bool onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) override;

    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

    int numberThread() const {
        return threadNumber();
    }
public:
    class BF16Creator {
    public:
        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op, Backend* backend) const = 0;
    };

    static bool addBF16Creator(OpType t, BF16Creator* ct);
};

#define REGISTER_BF16_OP_CREATOR(type, creator) \
    void ___##type##__##creator##__() { \
        BF16Backend::addBF16Creator(type, new creator); \
    }

} // namespace MNN

#endif /* BF16Backend_hpp */
