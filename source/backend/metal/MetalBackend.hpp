//
//  MetalBackend.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalBackend_hpp
#define MetalBackend_hpp

#include "Backend.hpp"
#include "MNN_generated.h"
#include "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

/** Metal backend */
class MetalBackend final : public Backend {
public:
    /** Metal execution creator */
    class Creator {
    public:
        /**
         * @brief create execution for given input, op on metal backend.
         * @param inputs    given input tensors.
         * @param op        given op.
         * @param backend   metal backend.
         * @return created execution if supported, NULL otherwise.
         */
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const = 0;
    };
    /**
     * @brief register creator for given op type.
     * @param type      given op type.
     * @param creator   registering creator.
     */
    static void addCreator(OpType type, Creator *creator);

public:
    MetalBackend();
    virtual ~MetalBackend();

    virtual bool onAcquireBuffer(const Tensor *Tensor, StorageType storageType) override;
    virtual bool onReleaseBuffer(const Tensor *Tensor, StorageType storageType) override;
    virtual bool onAllocateBuffer() override;
    virtual bool onClearBuffer() override;
    virtual void onCopyBuffer(const Tensor *srcTensor, const Tensor *dstTensor) const override;

    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op) override;
    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;
    virtual bool onWaitFinish() override;

public:
    /**
     * @brief get metal context object
     * @return metal context object pointer
     */
    void *context();

    /**
     * @brief copy buffer content to dest tensor
     * @param srcTensor source tensor
     * @param dstTensor destined tensor
     * @param encoder command encoder
     */
    virtual void onCopyBuffer(const Tensor *srcTensor, const Tensor *dstTensor,
                              id<MTLComputeCommandEncoder> encoder) const;

private:
    void *mContext = nil;
    std::map<void *, size_t> mStaticBuffers;
    std::map<void *, size_t> mDynamicBuffers;
    std::map<void *, size_t> mSeparatedBuffers;
    std::multimap<size_t, uint64_t> mReusableBuffers;
    mutable id<MTLBuffer> mHostBuffer = nil;

private:
    id<MTLBuffer> getHostBuffer(size_t size) const;
    void onCopyHostToDevice(const Tensor *src, const Tensor *dst) const;
    void onCopyDeviceToHost(const Tensor *src, const Tensor *dst) const;
    void onCopyDeviceToDevice(const Tensor *src, const Tensor *dst, id<MTLComputeCommandEncoder> encoder) const;
};

/** Metal creator register */
template <class T>
class MetalCreatorRegister {
public:
    /**
     * @brief initializer. register T creator for given op type.
     * @param type  given op type.
     */
    MetalCreatorRegister(OpType type) {
        T *test = new T;
        MetalBackend::addCreator(type, test);
    }
};
} // namespace MNN

#define REGISTER_METAL_OP_CREATOR(name, opType)     \
    void ___##name##__##opType##__() {              \
        MetalBackend::addCreator(opType, new name); \
    }

#endif /* MNN_METAL_ENABLED */
#endif /* MetalBackend_hpp */
