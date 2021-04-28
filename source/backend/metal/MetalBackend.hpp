//
//  MetalBackend.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalBackend_hpp
#define MetalBackend_hpp

#include "core/Backend.hpp"
#include "MNN_generated.h"
#include "MetalDefine.h"
#include <vector>

#if MNN_METAL_ENABLED
namespace MNN {
/** MetalRuntime */
class MetalRuntime : public Runtime {
public:
    friend class MetalBackend;
    class BufferAllocator {
    public:
        BufferAllocator(void* context);
        ~ BufferAllocator();
        id<MTLBuffer> alloc(size_t size, bool seperate = false);
        void release(id<MTLBuffer> buffer);
        void clear();
        float computeSizeInMB() const;
    private:
        std::map<id<MTLBuffer>, size_t> mAllocated;
        std::multimap<size_t, id<MTLBuffer>> mReusableBuffers;
        void* mContext = nullptr;
    };
    virtual float onGetMemoryInMB() override;

    MetalRuntime();
    virtual ~ MetalRuntime();
    virtual Backend* onCreate(const BackendConfig* config) const override;
    virtual void onGabageCollect(int level) override;
    void *context() const {
        return mContext;
    }
    id<MTLBuffer> getHostBuffer(size_t size) const;
private:
    void* mContext = nullptr;
    std::shared_ptr<BufferAllocator> mStatic;
    std::shared_ptr<BufferAllocator> mDynamic;
    mutable id<MTLBuffer> mHostBuffer = nullptr;
};

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

    class AutoBuffer {
    public:
        AutoBuffer(const MetalRuntime* runtime) {
            mRuntime = runtime;
        }
        ~ AutoBuffer();
        void reset(size_t length);
        id<MTLBuffer> buffer() const {
            return mBuffer;
        }
    private:
        const MetalRuntime* mRuntime = nullptr;
        id<MTLBuffer> mBuffer = nil;
    };
    const MetalRuntime* runtime() const {
        return mRuntime;
    }
public:
    MetalBackend(const MetalRuntime* runtime);
    virtual ~MetalBackend();

    virtual bool onAcquireBuffer(const Tensor *Tensor, StorageType storageType) override;
    virtual bool onReleaseBuffer(const Tensor *Tensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;
    virtual void onCopyBuffer(const Tensor *srcTensor, const Tensor *dstTensor) const override;

    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op) override;
    
    virtual void onResizeBegin() override;
    virtual void onResizeEnd() override;
    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;
    virtual std::pair<float, bool> onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                            const MNN::Op* op) override;

public:
    /**
     * @brief get metal context object
     * @return metal context object pointer
     */
    void *context() const;

    /**
     * @brief copy buffer content to dest tensor
     * @param srcTensor source tensor
     * @param dstTensor destined tensor
     * @param encoder command encoder
     */
    void onCopyBuffer(const Tensor *srcTensor, const Tensor *dstTensor,
                              id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> shape) const;

    void flushEncoder() const;
    id<MTLComputeCommandEncoder> encoder() const;
    void addOpEncoder(std::function<void(void)> opEncoder);
    
    bool isCommandEncoderSet() const;
    void setOpEncoder() const;
private:
    const MetalRuntime* mRuntime;
    std::vector<id<MTLBuffer>> mHoldBuffers;
    AutoBuffer mShapeH2D;
    AutoBuffer mShapeD2H;
    mutable bool mOpEncoderSet = false;
    mutable bool mOpFullSupport = true;
    mutable bool mFrameEncodeCache = false;

    std::vector<std::function<void(void)>> mOpEncoders;
    mutable id<MTLComputeCommandEncoder> mComputeEncoder = nil;

private:
    id<MTLBuffer> getHostBuffer(size_t size) const;
    void onCopyHostToDevice(const Tensor *src, const Tensor *dst) const;
    void onCopyDeviceToHost(const Tensor *src, const Tensor *dst) const;
    void onCopyDeviceToDevice(const Tensor *src, const Tensor *dst, id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> shape) const;
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
