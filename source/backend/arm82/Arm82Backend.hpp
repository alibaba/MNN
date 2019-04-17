//
//  Arm82Backend.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Arm82Backend_hpp
#define Arm82Backend_hpp

#include "Backend.hpp"
namespace MNN {
class Arm82Backend : public Backend {
public:
    virtual ~Arm82Backend() = default;
    Arm82Backend(int thread);
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
    virtual bool onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) override {
        return mCPUBackend->onAcquireBuffer(nativeTensor, storageType);
    }
    virtual bool onReleaseBuffer(const Tensor* nativeTensor, StorageType storageType) override {
        return mCPUBackend->onReleaseBuffer(nativeTensor, storageType);
    }
    virtual bool onAllocateBuffer() override {
        return mCPUBackend->onAllocateBuffer();
    }
    virtual bool onClearBuffer() override {
        return mCPUBackend->onClearBuffer();
    }

    virtual void onExecuteBegin() const override {
        return mCPUBackend->onExecuteBegin();
    }
    virtual void onExecuteEnd() const override {
        return mCPUBackend->onExecuteEnd();
    }

    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override {
        mCPUBackend->onCopyBuffer(srcTensor, dstTensor);
    }

    int numberThread() const {
        return mNumberThread;
    }

private:
    std::shared_ptr<Backend> mCPUBackend;
    int mNumberThread;
};
} // namespace MNN

#endif /* Arm82Backend_hpp */
