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
#include <atomic>
#include "MNN_generated.h"
#include "MetalDefine.h"
#include <MNN/ErrorCode.hpp>
#include <vector>
#include <queue>
#include <unordered_map>
//#include "MNNMetalContext.h"
#include "MetalCache_generated.h"
using namespace MetalCache;

#if MNN_METAL_ENABLED
namespace MNN {

class MetalConvolution1x1; // forward declaration for Gate/Up fusion

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
    bool supportSimdGroupReduce() {
        return mSimdGroupReduce;
    }
    bool supportSimdGroupMatrix() {
        return mSimdGroupMatrix;
    }
    bool supportTensorOps() {
        return mTensorOps;
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
    std::map<std::string, std::vector<std::pair<std::vector<uint32_t>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, uint32_t>>>>& getTunedThreadGroupVec() {
        return mTunedThreadGroupVec;
    }
    virtual Backend *onCreate(const BackendConfig* config, Backend* origin) const override;
    virtual void onGabageCollect(int level) override;
    virtual CompilerType onGetCompilerType() const override {
        return Compiler_Loop;
    }
    virtual float onGetMemoryInMB() override;

    virtual std::pair<const void*, size_t> onGetCache() override;
    virtual bool onSetCache(const void* buffer, size_t size) override;

    static MetalRuntime* create(const Backend::Info& info);
    virtual void onMaskOpReady(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const MNN::Op* op) override;
    virtual bool onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                        const MNN::Op* op, Runtime::OpInfo& dstInfo) const override;
    SingleBufferWithAllocator* buffer(int index) const {
        return &mDynamic[index];
    }
    BufferAllocator* createDynamicAllocator(int index, bool secondResize) const;
    mutable id<MTLCommandBuffer> _waiting = nil;

    size_t maxThreadSize() const {
        return mMaxThreadSize;
    }

private:
    MetalRuntime(void* context);
    void* mContext = nullptr;
    mutable std::shared_ptr<EagerBufferAllocator> mStaticAllocator;
    mutable std::shared_ptr<EagerBufferAllocator> mStaticAllocatorRaw;
    mutable std::shared_ptr<EagerBufferAllocator> mStaticAllocatorMMap;

    mutable std::vector<SingleBufferWithAllocator> mDynamic;
    MetalTuneLevel mTuneLevel = Wide;
    std::map<std::pair<std::string, std::vector<uint32_t>>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, uint32_t>> mTunedThreadGroup;
    std::map<std::string, std::vector<std::pair<std::vector<uint32_t>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, uint32_t>>>> mTunedThreadGroupVec;

private:
    id<MTLCommandQueue> mQueue = nil;
    bool mUserSync = false;
    std::vector<uint8_t> mBuffer;
    const void* mCacheOutside = nullptr;
    size_t mCacheOutsideSize = 0;
    TunedInfo* mTunedInfo;
    BackendConfig mDefaultConfig;
    mutable std::map<std::vector<std::string>, id<MTLComputePipelineState>> mCachePipeine;
private:
    bool mSimdGroupReduce;
    bool mSimdGroupMatrix;
    bool mTensorOps;
    size_t mMaxThreadSize;
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
    static void setMem(const MemChunk& chunk, id<MTLComputeCommandEncoder> encoder, int index);
    static uint8_t* getMemPtr(const MemChunk& chunk);
    static void setBuffer(id<MTLBuffer> buffer, int offset, id<MTLComputeCommandEncoder> encoder, int index);
    static std::pair<id<MTLBuffer>, int> getBuffer(const MNN::Tensor* tensor);
    size_t getTensorSizeInBytes(const Tensor* tensor) const;
    virtual bool onSelectDynamicAllocator(int index, int maxIndex) override;
    id<MTLBuffer> getHostBuffer(size_t size) const;
    id<MTLBuffer> getConstBuffer(size_t size) const;
    void returnConstBuffer(id<MTLBuffer> buffer) const;
    id<MTLComputePipelineState> makeComputePipelineWithSourceOption(const char* csource, const char* cname, MTLCompileOptions *options) const;
public:
    MetalBackend(const MetalRuntime* runtime, bool usefp16AsFp32, BackendConfig::MemoryMode mode);
    virtual ~MetalBackend();
    virtual Runtime* getRuntime() override {
        return (Runtime*)mRuntime;
    }
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
    
    BufferAllocator* getBufferPool() const;
    EagerBufferAllocator *getStaticBufferPool() const {
        return mRuntime->mStaticAllocator.get();
    }
    id<MTLCommandBuffer> getCommandBufferForBufferCopy() const;

    bool isCmdBufferCommit();
#if MNN_METAL_OP_PROFILE
    // Register an execution's op name for per-op GPU profiling (called at onCreate).
    void profileRegisterOp(const Execution* exe, const std::string& name) const;
    // Mark the currently executing op so the next committed command buffer is attributed to it.
    void profileMarkOp(const Execution* exe) const;
    // Append a kernel-variant subtag to the current op profile name. Called from onEncode
    // after the pipeline has been selected so profile rows can distinguish kernels.
    // Example: OpType="Convolution", subtag="gemm_32x64_split_k" -> "Convolution/gemm_32x64_split_k"
    void setProfileSubtag(const std::string& subtag) const;
#endif
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
    BackendConfig::MemoryMode getMemoryMode() const {
        return mMemoryMode;
    }
    bool isSupportTensorApi() const {
        return mSupportTensorApi;
    }

    // Gate/Up projection fusion: register Conv1x1 execution for its output tensor
    void registerConv1x1ForOutput(const Tensor* output, MetalConvolution1x1* exe) {
        mOutputToConv1x1[output] = exe;
    }
    MetalConvolution1x1* findConv1x1ForOutput(const Tensor* output) const {
        auto it = mOutputToConv1x1.find(output);
        return it != mOutputToConv1x1.end() ? it->second : nullptr;
    }

    // QKV fusion: register Conv1x1 grouped by input tensor for later matching
    struct QKVCandidate {
        MetalConvolution1x1* conv;
        const Tensor* output;
        int outputChannel;
    };
    void registerConv1x1ForQKV(const Tensor* input, MetalConvolution1x1* conv, const Tensor* output, int outputChannel) {
        mInputToConv1x1Group[input].push_back({conv, output, outputChannel});
    }
    void matchQKVFusions();  // called in onResizeEnd

    // LayerNorm+Conv1x1 fusion: register LN by its normalized output for matching
    struct LayerNormFusionInfo {
        const Tensor* hiddenInput;      // LN inputs[1] → Conv1x1's in (buffer 0)
        const Tensor* residualInput;    // LN inputs[0] → ln_residual_in (buffer 20)
        const Tensor* residualOutput;   // LN outputs[0] → ln_residual_out (buffer 22)
        std::shared_ptr<Tensor> gamma;  // LN gamma tensor (float4 data)
        float eps;
        bool* fusedFlag;                // points to MetalLayerNorm::mIsFused
    };
    void registerLayerNorm(const Tensor* normalizedOutput, const LayerNormFusionInfo& info) {
        mLayernormMap[normalizedOutput] = info;
    }
    void matchLNFusions();  // called after matchQKVFusions in onResizeEnd

    void clearConv1x1Map() {
        mOutputToConv1x1.clear();
        mInputToConv1x1Group.clear();
        mLayernormMap.clear();
    }
private:
    BackendConfig::MemoryMode mMemoryMode;
    bool mSupportTensorApi = false;
    // Gate/Up fusion: maps output tensor to its Conv1x1 execution
    std::unordered_map<const Tensor*, MetalConvolution1x1*> mOutputToConv1x1;
    // QKV fusion: maps input tensor to group of Conv1x1 candidates
    std::unordered_map<const Tensor*, std::vector<QKVCandidate>> mInputToConv1x1Group;
    std::unordered_map<const Tensor*, LayerNormFusionInfo> mLayernormMap;
private:
    MetalRuntimeAllocator::MetalBufferAlloc mEmptyMem;
    id<MTLCommandBuffer> getCommandBufferForNet() const;
    id<MTLComputeCommandEncoder> encoder_net() const;
    mutable id<MTLCommandBuffer> _commandBuffer = nil;
    mutable std::queue<id<MTLBuffer>> mHoldBuffers;

    id<MTLCommandQueue> _commandQueue;

    const MetalRuntime* mRuntime;
    mutable NSUInteger mEncoderCount = 0;

    mutable id<MTLComputeCommandEncoder> mComputeEncoder = nil;
    std::shared_ptr<BufferAllocator> mBufferPool;
    std::shared_ptr<BufferAllocator> mBufferPoolShapeImmutable;
    std::atomic<bool> mGPUEnabledSwitch;
    id<NSObject> mForegroundObserver;
    id<NSObject> mBackgroundObserver;

private:
    void _resetDynamicMemory() const;
    CopyPipeline _makeCopyInfo(const Tensor *src, const Tensor *dst, id<MTLBuffer> shape, int castType) const;
    void setUpGPUEnabledSwitch();
    void removeNotificationsObservers();

    mutable id<MTLBuffer> mHostBuffer = nullptr;
    // hostmask: 0: no host, 1: src is host, 2: dst is host
    void onCopyDeviceToDevice(const Tensor *src, const Tensor *dst, id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> shape, int hostmask = 0) const;
    bool mUseFloatAsFp16;
    bool mIsIphone = false;
    BufferAllocator* mCurrentAllocator = nullptr;
    std::shared_ptr<BufferAllocator> mExecutionBufferPool;
#if MNN_METAL_OP_PROFILE
    mutable std::string mCurProfileName;
#endif

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

#if MNN_METAL_OP_PROFILE
// Register a cloned execution's op name for per-op GPU profiling.
// Must be called inside each MetalExecution subclass's onClone (op carries the type).
#define MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, dstExe)                                       \
    do {                                                                                        \
        if ((dstExe) != nullptr && (op) != nullptr) {                                           \
            static_cast<MNN::MetalBackend*>(bn)->profileRegisterOp((dstExe),                     \
                                                                   MNN::EnumNameOpType((op)->type())); \
        }                                                                                       \
    } while (0)
#else
#define MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, dstExe)
#endif

#endif /* MNN_METAL_ENABLED */
#endif /* MetalBackend_hpp */