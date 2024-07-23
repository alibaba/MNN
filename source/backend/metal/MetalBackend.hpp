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
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "MetalDefine.h"
#include <MNN/ErrorCode.hpp>
#include <vector>
#include <queue>
//#include "MNNMetalContext.h"
#include "MetalCache_generated.h"
using namespace MetalCache;

#if MNN_METAL_ENABLED
namespace MNN {

/** MetalRuntime */
enum MetalTuneLevel {Never = 0, Heavy = 1, Wide = 2, Normal = 3, Fast = 4};

struct TunedInfo;
class MetalRuntime : public Runtime {
public:
    friend class MetalBackend;
    virtual ~ MetalRuntime();
    
    void *context() const {
        return mContext;
    }

    void setGpuMode(const int cl_mode_num);
    void setCommandQueue(id<MTLCommandQueue> queue, bool userSync);
    id<MTLCommandQueue> getCommandQueue() const {
        return mQueue;
    }
    bool userSync() const {
        return mUserSync;
    }
    
    std::pair<const void*, size_t> makeCache(TunedInfo* info);
    bool setCache(std::pair<const void*, size_t> cache);
    id<MTLComputePipelineState> findPipeline(const std::vector<std::string>& keys) const;
    void insertPipeline(const std::vector<std::string>& keys, id<MTLComputePipelineState> pipeline) const;
    MetalTuneLevel getTuneLevel() {
        return mTuneLevel;
    }
    std::map<std::pair<std::string, std::vector<uint32_t>>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>,  uint32_t>>& getTunedThreadGroup() {
        return mTunedThreadGroup;
    };
    virtual Backend *onCreate(const BackendConfig* config) const override;
    virtual void onGabageCollect(int level) override;
    virtual CompilerType onGetCompilerType() const override {
        return Compiler_Loop;
    }
    virtual float onGetMemoryInMB() override;

    virtual std::pair<const void*, size_t> onGetCache() override;
    virtual bool onSetCache(const void* buffer, size_t size) override;

    static MetalRuntime* create(const Backend::Info& info, id<MTLDevice> device);
    virtual void onMaskOpReady(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const MNN::Op* op) override;
    virtual bool onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                        const MNN::Op* op, Runtime::OpInfo& dstInfo) const override;
private:
    MetalRuntime(void* context);
    void* mContext = nullptr;
    std::shared_ptr<EagerBufferAllocator> mStatic;
    MetalTuneLevel mTuneLevel = Wide;
    std::map<std::pair<std::string, std::vector<uint32_t>>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, uint32_t>> mTunedThreadGroup;

private:
    id<MTLCommandQueue> mQueue = nil;
    bool mUserSync = false;
    std::vector<uint8_t> mBuffer;
    const void* mCacheOutside = nullptr;
    size_t mCacheOutsideSize = 0;
    TunedInfo* mTunedInfo;
    BackendConfig mDefaultConfig;
    mutable std::map<std::vector<std::string>, id<MTLComputePipelineState>> mCachePipeine;
};


class MetalRuntimeAllocator : public BufferAllocator::Allocator {
public:
    class MetalBufferAlloc {
    public:
        MetalBufferAlloc(id<MTLBuffer> buffer) {
            mBuffer = buffer;
        }
        id<MTLBuffer> getBuffer() {
            return mBuffer;
        }
        ~MetalBufferAlloc(){};
    private:
        id<MTLBuffer> mBuffer;
    };
    
    MetalRuntimeAllocator(id<MTLDevice> device): mDevice(device) {
        // Do nothing
    }
    virtual ~ MetalRuntimeAllocator() = default;
    virtual MemChunk onAlloc(size_t size, size_t align) override;
    virtual void onRelease(MemChunk ptr) override;
    
private:
    id<MTLDevice> mDevice;
};

/** Metal backend */
class MetalBackend : public Backend {
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
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *> &outputs) const = 0;
    };
    /**
     * @brief register creator for given op type.
     * @param type      given op type.
     * @param creator   registering creator.
     */
    static void addCreator(OpType type, Creator *creator);
    static void setTensor(const MNN::Tensor* tensor, id<MTLComputeCommandEncoder> encoder, int index);
    static std::pair<id<MTLBuffer>, int> getBuffer(const MNN::Tensor* tensor);
    size_t getTensorSizeInBytes(const Tensor* tensor) const;
    virtual bool onSelectDynamicAllocator(int index, int maxIndex) override;
    id<MTLBuffer> getHostBuffer(size_t size) const;
    id<MTLBuffer> getConstBuffer(size_t size) const;
    void returnConstBuffer(id<MTLBuffer> buffer) const;
    id<MTLComputePipelineState> makeComputePipelineWithSourceOption(const char* csource, const char* cname, MTLCompileOptions *options) const;
public:
    MetalBackend(std::shared_ptr<EagerBufferAllocator> staticMem, const MetalRuntime* runtime, bool usefp16AsFp32);
    virtual ~MetalBackend();
    const MetalRuntime* runtime() const {
        return mRuntime;
    }
    
    virtual Backend::MemObj* onAcquire(const Tensor *Tensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;
    virtual void onCopyBuffer(const Tensor *srcTensor, const Tensor *dstTensor) const override;

    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op) override;
    
    virtual void onResizeBegin() override;
    virtual ErrorCode onResizeEnd() override;
    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;
    virtual int onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) override;
    virtual bool onGetTensorInfo(const Tensor* tensor, void* dstInfo) override;

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
    id<MTLComputeCommandEncoder> encoder_for_net() const;
    void addOpEncoder(std::function<void(void)> opEncoder);
    
    bool isCommandEncoderSet();
    
    BufferAllocator* getBufferPool() const;
    EagerBufferAllocator *getStaticBufferPool() const {
        return mStaticBufferPool.get();
    }

    bool isCmdBufferCommit();
    bool isIphone(){
        return mIsIphone;
    }
    
    void commit() const;
    void commit_net() const;
    void wait() const;
    id<MTLCommandQueue> queue() const {
        return _commandQueue;
    }
    bool useFp16InsteadFp32() const {
        return mUseFloatAsFp16;
    }
    struct CopyPipeline {
        id<MTLComputePipelineState> pipeline;
        id<MTLBuffer> shape;
        MTLSize localSize;
        MTLSize groupSize;
    };
private:
    MetalRuntimeAllocator::MetalBufferAlloc mEmptyMem;
    id<MTLCommandBuffer> getCommandBufferForBufferCopy() const;
    id<MTLCommandBuffer> getCommandBufferForNet() const;
    id<MTLComputeCommandEncoder> encoder_net() const;
    mutable id<MTLCommandBuffer> _commandBuffer = nil;
    mutable id<MTLCommandBuffer> _commandBuffer_net = nil;
    mutable id<MTLCommandBuffer> _waiting = nil;
    mutable std::queue<id<MTLBuffer>> mHoldBuffers;

    id<MTLCommandQueue> _commandQueue;

    const MetalRuntime* mRuntime;
    id<MTLBuffer> mShapeH2D;
    id<MTLBuffer> mShapeD2H;
    mutable NSUInteger mEncoderCount = 0;
    mutable bool mOpEncoderSet = false;//whether has set encoder
    mutable bool mSupportDeferEncode = true;
    mutable bool mFrameEncodeCache = false;

    std::vector<std::function<void(void)>> mOpEncoders;
    mutable id<MTLComputeCommandEncoder> mComputeEncoder = nil;
    std::shared_ptr<BufferAllocator> mBufferPool;
    std::shared_ptr<BufferAllocator> mBufferPoolShapeImmutable;
    std::shared_ptr<EagerBufferAllocator> mStaticBufferPool;

private:
    CopyPipeline _makeCopyInfo(const Tensor *src, const Tensor *dst, id<MTLBuffer> shape, int castType) const;

    mutable id<MTLBuffer> mHostBuffer = nullptr;
    // hostmask: 0: no host, 1: src is host, 2: dst is host
    void onCopyDeviceToDevice(const Tensor *src, const Tensor *dst, id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> shape, int hostmask = 0) const;
    bool mUseFloatAsFp16;
    bool mIsIphone = false;
    BufferAllocator* mCurrentAllocator = nullptr;
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

#define REGISTER_METAL_OP_TRANSFORMER_CREATOR(name, opType)     \
    void ___##name##__##opType##__() {              \
        MetalBackend::addCreator(opType, new name); \
    }

#endif /* MNN_METAL_ENABLED */
#endif /* MetalBackend_hpp */
