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
#include "MetalEnv.hpp"
#include <MNN/ErrorCode.hpp>
#include <vector>
#include <queue>
#include <set>
#include <unordered_map>
//#include "MNNMetalContext.h"
#include "MetalCache_generated.h"
using namespace MetalCache;

#if MNN_METAL_ENABLED
namespace MNN {

class MetalConvolution1x1; // forward declaration for Gate/Up fusion
class MetalRaster;         // forward declaration for the gated-norm fold leader

// Compile with -DMNN_SESSION_CPU_TRACE: cumulative CPU-side timers for the op encode path
// and command-buffer waits, printed at process exit. Quantifies the
// encode-reuse / ICB leverage ceiling. Not compiled in production builds.
#ifdef MNN_SESSION_CPU_TRACE
struct MetalCpuTrace {
    // Pure onEncode / encoder_for_net (mid-execution flush split off below).
    std::atomic<uint64_t> encodeNs{0};
    std::atomic<uint64_t> encodeOps{0};
    // flushEncoder + commit_net inside a Execution (isCmdBufferCommit path).
    std::atomic<uint64_t> commitNs{0};
    std::atomic<uint64_t> commitCalls{0};
    // Blocking waits on the last-committed command buffer.
    std::atomic<uint64_t> waitNs{0};
    std::atomic<uint64_t> waitCalls{0};
    // per-site wait attribution: 0=resizeFence 1=d2h 2=h2d 3=onSync
    std::atomic<uint64_t> waitSiteNs[4]{};
    std::atomic<uint64_t> waitSiteCalls[4]{};
    // GPU-side utilization: per-command-buffer busy time and inter-buffer gaps
    std::atomic<uint64_t> gpuBusyNs{0};
    std::atomic<uint64_t> gpuGapNs{0};
    std::atomic<uint64_t> gpuBuffers{0};
    std::atomic<double> gpuPrevEnd{0.0};
    ~MetalCpuTrace();
};
MetalCpuTrace& metalCpuTrace();
#endif // MNN_SESSION_CPU_TRACE

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
    bool supportSimdGroupReduce() const {
        return mSimdGroupReduce;
    }
    bool supportSimdGroupMatrix() {
        return mSimdGroupMatrix;
    }
    bool supportTensorOps() {
        return mTensorOps;
    }
    bool preferInShaderPrefillDequant() {
        return mPreferInShaderPrefillDequant;
    }
    // M64 outer-dequant GEMM tile tier (see MetalConvolution1x1). True only on
    // M4-class Macs, resolved from MTLDevice.architecture.name at runtime init.
    bool preferM64Gemm() {
        return mPreferM64Gemm;
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
    // A nil pipeline is remembered as a compile failure rather than dropped, so
    // callers can skip retrying a shader that cannot build. Without this a
    // failing optional fusion re-compiles from source on every resize.
    void insertPipeline(const std::vector<std::string>& keys, id<MTLComputePipelineState> pipeline) const;
    bool pipelineCompileFailed(const std::vector<std::string>& keys) const;
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
    virtual int onGetRuntimeStatus(RuntimeStatus statusEnum) const override;
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
    mutable std::set<std::vector<std::string>> mFailedPipeline;
private:
    bool mSimdGroupReduce;
    bool mSimdGroupMatrix;
    bool mTensorOps;
    bool mPreferInShaderPrefillDequant = false;
    bool mPreferM64Gemm = false;
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
    // queued host->device upload staging ring (see onCopyBuffer h2d path)
    id<MTLBuffer> acquireUploadStaging(size_t size) const;
    void markUploadStagingUse(id<MTLBuffer> staging, id<MTLCommandBuffer> cmd) const;
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
    // Split an op into sub-passes for profiling. Counter mode: ends the current
    // encoder (cheap, stays in the same command buffer). Legacy mode: commits a
    // command buffer per sub-pass. Returns a fresh encoder tagged with `subtag`.
    id<MTLComputeCommandEncoder> profileNextSubpass(const std::string& subtag) const;
    // Per-op encoder boundary (called by MetalExecution after onEncode in counter mode).
    void profileOpEncoded() const;
    // Called by ops that encode NOTHING (fused followers' early return): an empty
    // encoder's stage-boundary timestamps measure scheduling gaps, not GPU work,
    // so drop its sample instead of polluting the profile.
    void profileDropCurrentSample() const {
        mProfileCurSampleIndex = -1;
    }
    bool profileUseCounters() const {
        return mProfileCounterMode;
    }
#endif
    bool isIphone(){
        return mIsIphone;
    }
    
    void commit() const;
    void commit_net() const;
    void wait(int traceSite = -1) const;
    // wait only for this backend's own in-flight command buffer (see member note)
    void waitOwnInflight() const;
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
    // matmul2d input cooperative tensors (single-simdgroup scope + per-element
    // operator[]). Required by fused attention, where the QK destination must
    // become the PV left operand without going through memory. Probed
    // separately from mSupportTensorApi because MNN's other tensor kernels only
    // need a destination cooperative tensor at execution_simdgroups<4>.
    bool isSupportTensorCoopInput() const {
        return mSupportTensorCoopInput;
    }

    // Export-time fused projection ops (FusedLinear) establish their
    // own leader/follower wiring from the exported member order. They must do it
    // after the allocator's compute(), because the setup re-homes follower
    // outputs to STATIC, so they register here and onResizeEnd calls them back.
    class FusedProjFusionHost {
    public:
        virtual ~FusedProjFusionHost() = default;
        virtual void setupFusion() = 0;
    };
    void registerFusedProj(FusedProjFusionHost* host) {
        mFusedProjs.push_back(host);
    }
    // True when two tensors share backing memory. Any fusion that folds a
    // producer into a consumer must check that the producer's inputs were not
    // reused as the consumer's outputs: legal under the unfused schedule, but a
    // read/write race once both live in one dispatch.
    // Not static: the span of a tensor is getTensorSizeInBytes(), which needs the
    // backend's fp16 mode, and an under-reported span misses real aliases.
    bool tensorsOverlap(const Tensor* a, const Tensor* b) const;

    // LinearAttention gate/beta fold, declared by the exported
    // LinearAttentionParam (gate_fold). The op fills the request from its param;
    // the backend only re-homes the raw a/b inputs.
    struct LinearAttnFoldRequest {
        const Tensor* gate = nullptr;   // LinearAttention inputs[1]
        const Tensor* beta = nullptr;   // LinearAttention inputs[2]
        int numHeads = 0;
        const Tensor* rawA = nullptr;   // bound instead of gate
        const Tensor* rawB = nullptr;   // bound instead of beta
        std::vector<float> gateCoef;    // -exp(A_log)[h]
        std::vector<float> gateBias;    // dt_bias[h]
        bool gateFolded = false;
        bool betaFolded = false;
        // Set by LinearAttention when the model carries gate_fold in its param.
        bool exportFold = false;
    };
    void registerLinearAttnFold(LinearAttnFoldRequest* req) {
        mLinearAttnFolds.push_back(req);
    }
    ErrorCode applyLinearAttnGateFolds(); // called in onResizeEnd

    void clearConv1x1Map() {
        mFusedProjs.clear();
        mLinearAttnFolds.clear();
    }
private:
    BackendConfig::MemoryMode mMemoryMode;
    bool mSupportTensorApi = false;
    bool mSupportTensorCoopInput = false;
    // Export-time fused projections, in resize order
    std::vector<FusedProjFusionHost*> mFusedProjs;
    std::vector<LinearAttnFoldRequest*> mLinearAttnFolds;
private:
    MetalRuntimeAllocator::MetalBufferAlloc mEmptyMem;
    id<MTLCommandBuffer> getCommandBufferForNet() const;
    id<MTLComputeCommandEncoder> encoder_net() const;
    mutable id<MTLCommandBuffer> _commandBuffer = nil;
    // Per-backend fence: the last command buffer THIS backend committed.
    // onResizeBegin only needs to wait for our own in-flight work before
    // resetting our allocator; draining the shared runtime's last commit
    // (which may belong to another module's backend) serializes decode.
    mutable id<MTLCommandBuffer> mLastOwnCommandBuffer = nil;
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
    // staging ring for queued host->device uploads: each slot remembers the
    // command buffer that consumes it so reuse never races in-flight GPU reads
    struct UploadStagingSlot {
        id<MTLBuffer> buffer = nil;
        id<MTLCommandBuffer> lastUse = nil;
    };
    mutable std::vector<UploadStagingSlot> mUploadStagingRing;
    // hostmask: 0: no host, 1: src is host, 2: dst is host
    void onCopyDeviceToDevice(const Tensor *src, const Tensor *dst, id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> shape, int hostmask = 0) const;
    bool mUseFloatAsFp16;
    bool mIsIphone = false;
    BufferAllocator* mCurrentAllocator = nullptr;
    std::shared_ptr<BufferAllocator> mExecutionBufferPool;
#if MNN_METAL_OP_PROFILE
    mutable std::string mCurProfileName;
    // MTLCounterSampleBuffer-based per-encoder GPU timing (accurate absolute
    // numbers, no per-op command buffer commit). Falls back to legacy
    // whole-command-buffer timing when stage-boundary sampling is unsupported
    // or MNN_METAL_OP_PROFILE_LEGACY=1.
    bool mProfileCounterMode = false;
    struct ProfilePendingSample {
        int index;         // startOfEncoderSampleIndex; end = index + 1
        std::string name;
    };
    mutable id<MTLCounterSampleBuffer> mProfileSampleBuffer = nil;
    mutable int mProfileSampleCursor = 0;
    mutable int mProfileCurSampleIndex = -1;  // sample index of the live encoder
    mutable std::vector<ProfilePendingSample> mProfilePendingSamples;
    // Sample buffers filled up mid-command-buffer, waiting for resolution at commit.
    struct ProfileSealedBuffer {
        id<MTLCounterSampleBuffer> buffer;
        int usedCount;
        std::vector<ProfilePendingSample> samples;
    };
    mutable std::vector<ProfileSealedBuffer> mProfileSealedBuffers;
    id<MTLCounterSampleBuffer> profileAcquireSampleBuffer() const;
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