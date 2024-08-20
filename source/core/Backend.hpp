//
//  Backend.hpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Backend_hpp
#define Backend_hpp

#include <MNN/MNNForwardType.h>
#include <MNN/ErrorCode.hpp>
#include <map>
#include "Command.hpp"
#include "NonCopyable.hpp"
#include "BufferAllocator.hpp"
#include <future>
#include <atomic>

namespace MNN {

struct Op;
class Execution;

class Runtime;
class Backend;
struct RuntimeHint {
    // 0: Defer, 1: Eager
    int memoryAllocatorType = 0;
    int winogradMemoryUsed = 3;
    
    // 0-100, 50 means litter core has 50% capacity of large core
    int cpuDecreaseRate = 50;
    int dynamicQuantOption = 0;

    // 0: Do not quantize kvcache, just store float
    // 1: Only quantize key cache, use int8 asymmetric quantization 
    // 2: Only quantize value cache, use fp8 quantization
    // 3: quantize both key and value cache as described above
    int kvcacheQuantOption = 0;
};
/** abstract backend */
class Backend : public NonCopyable {

public:
    /** info used to create backend */
    struct Info {
        /** forward type. */
        MNNForwardType type = MNN_FORWARD_CPU;
        /** numThread for CPU . number of threads.  gpuMode for GPU only. tuning/memory Mode setting. */
        union {
            int numThread = 4;
            int gpuMode;
        };
        /** user data. */
        BackendConfig* user = NULL;
        enum Mode {
            // The Op will be run in execution->onExecute
            DIRECT = 0,

            // The Op will be recorded. Run in onExecuteBegin and Wait in onExecuteEnd
            INDIRECT = 1
        };
        Mode mode = DIRECT;
    };

    /** backend buffer storage type */
    enum StorageType {
        /**
         use NOT reusable memory.
         - allocates memory when `onAcquireBuffer` is called.
         - releases memory when `onReleaseBuffer` is called or when the backend is deleted.
         - do NOTHING when `onClearBuffer` is called.
         */
        STATIC,
        /**
         use reusable memory.
         - allocates or reuses memory when `onAcquireBuffer` is called. prefers reusing.
         - collects memory for reuse when `onReleaseBuffer` is called.
         - releases memory when `onClearBuffer` is called or when the backend is deleted.
         */
        DYNAMIC,
        /**
         use NOT reusable memory.
         - allocates memory when `onAcquireBuffer` is called.
         - do NOTHING when `onReleaseBuffer` is called.
         - releases memory when `onClearBuffer` is called or when the backend is deleted.
         */
        DYNAMIC_SEPERATE
    };

public:
    /**
     * @brief initializer.
     * @param type  forward type.
     */
    Backend(MNNForwardType type) : mType(type) {
        // nothing to do
    }

    /**
     * @brief deinitializer.
     */
    virtual ~Backend() = default;

public:

    /**
     * @brief create execution for op with input and output tensors.
     * @param inputs    input tensors.
     * @param outputs   output tensors.
     * @param op        given op.
     * @return created execution if op is supported, nullptr otherwise.
     */
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) = 0;

    /**
     * @brief callback before resize ops.
     */
    virtual void onResizeBegin() {
        // nothing to do
    }
    /**
     * @brief callback after resize ops.
     */
    virtual ErrorCode onResizeEnd() = 0;

    /**
     * @brief callback before executing ops.
     */
    virtual void onExecuteBegin() const = 0;
    /**
     * @brief callback after executing ops.
     */
    virtual void onExecuteEnd() const = 0;

    virtual const Runtime* getRuntime() {
        return nullptr;
    }
    
    /**
     * @brief allocate buffer of tensor for given storage type.
     * @param tensor        buffer provider.
     * @param storageType   buffer storage type.
     * @return success or not.
     */
    MNN_PUBLIC bool onAcquireBuffer(const Tensor* tensor, StorageType storageType);

    /**
     * @brief release buffer of tensor for given storage type.
     * @param tensor        buffer provider.
     * @param storageType   buffer storage type.
     * @return success or not.
     */
    MNN_PUBLIC bool onReleaseBuffer(const Tensor* tensor, StorageType storageType);

    class MemObj : public RefCount {
    public:
        MemObj() {}
        virtual ~ MemObj() {}
        virtual MemChunk chunk() { return MemChunk(); }
    };
    /**
     * @brief allocate buffer of tensor for given storage type.
     * @param tensor        buffer provider.
     * @param storageType   buffer storage type.
     * @return MemObj for release, if failed, return nullptr.
     */
    virtual MemObj* onAcquire(const Tensor* tensor, StorageType storageType) = 0;
    
    virtual bool onSelectDynamicAllocator(int index, int maxIndex) {
        return false;
    }
    /**
     * @brief get buffer from tensor directly
     * @param tensor        buffer provider.
     * @return support or not
     */
    virtual bool onGetTensorInfo(const Tensor* tensor, void* dstInfo) {
        return false;
    }

    /**
     * @brief clear all dynamic buffers.
     * @return success or not.
     */
    virtual bool onClearBuffer() = 0;

    /**
     * @brief copy buffer from tensor to tensor.
     * @param srcTensor source buffer provider.
     * @param dstTensor dest buffer provider.
     */
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const = 0;

public:
    /**
     * @brief get forward type.
     * @return forward type.
     */
    inline MNNForwardType type() const {
        return mType;
    }

public:
    /**
     * @brief get Gpu Tensor map host ptr/ unmap
     */
    virtual void* onMapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* srcTensor) {
        return nullptr;
    }

    virtual bool onUnmapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* dstTensor, void* mapPtr) {
        return false;
    }

    virtual int onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) {
        return 0;
    }

private:
    const MNNForwardType mType;
};

/** Each backend belong to a runtime*/
class Runtime : public NonCopyable {
public:
    /**
     Origin Op -> (Compiler) -> New Op -> Backend
     Default use Compiler_Geometry, Origin Op -> Compiler_Geometry -> Little Op
     For serveral Backend, we can't use Geometry to decompose origin op, then it set Compiler_Origin
     */
    enum CompilerType {
        Compiler_Geometry = 0,
        Compiler_Origin = 1,
        Compiler_Loop = 2,
    };

    enum AllocatorType {
        Allocator_Defer = 0,
        Allocator_Eager = 1,
    };
    void setRuntimeHint(const RuntimeHint& hint) {
        mHint = hint;
    }
    const RuntimeHint& hint() const {
        return mHint;
    }

    virtual CompilerType onGetCompilerType() const {
        return Compiler_Loop;
    }

    virtual ~Runtime() = default;
    /**
     @brief create backend
     @return created backend
     */
    virtual Backend* onCreate(const BackendConfig* config = nullptr) const = 0;

    /**
     @brief reset runtime
     */
    virtual void onReset(int numberThread, const BackendConfig* config) {
        // Do nothing
    }

    /**
     @brief clear unuseful resource
     @param level clear level: 0 - 100, bigger mean clear more, smaller mean cache more
     */
    virtual void onGabageCollect(int level) = 0;

    /**
     @brief Measure the memory it used in MB
     */
    virtual float onGetMemoryInMB() {
        return 0.0f;
    }

    // If buffer is not nullptr, try copy cache, else delete cache
    virtual bool onSetCache(const void* buffer, size_t size) {
        //default cache valid, avoid being reset
        return true;
    }

    virtual std::pair<const void*, size_t> onGetCache() {
        return std::make_pair(nullptr, 0);
    }
    virtual int onGetRuntimeStatus(RuntimeStatus statusEnum) const {
        return 0;
    }
    // If the info user set can't be match by runtime, return false and set real info
    virtual bool onCheckInfo(Backend::Info& info) const {
        return true;
    }
    struct OpInfo {
        bool initCostLong;
        float exeutionCost; // In ms
        float initCost; // In ms
    };
    /**
     * @brief measure the cost for op with input and output tensors.
     * @param inputs    input tensors.
     * @param outputs   output tensors.
     * @param op        given op.
     * @param dstInfo   the Info for write.
     * @return support the op or not;
     */
    virtual bool onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                             const MNN::Op* op, OpInfo& dstInfo) const {
        return true;
    }

    // FIXME: Temply use to mask cache valid, in future will delete
    virtual void onMaskOpReady(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const MNN::Op* op) {
        // Do nothing
    }
    // FIXME: Temply used, in future will refract
    std::atomic_bool mCancelled = ATOMIC_VAR_INIT(false);
    MNN_PUBLIC bool hasAsyncWork() const;
    void setAsyncWork(std::future<int>&& future);
    MNN_PUBLIC void waitAsyncWork();
private:
    std::future<int> mFuture;
    RuntimeHint mHint;
};

/** abstract Runtime register */
class RuntimeCreator {
public:
    /**
     @brief initializer.
     */
    virtual ~RuntimeCreator() = default;

    virtual Runtime* onCreate(const Backend::Info& info) const = 0;
    /**
     @brief Turn info to supported.
     @param info    info to valid.
     @return success or not
     */
    virtual bool onValid(Backend::Info& info) const {
        info.mode = Backend::Info::DIRECT;
        return true;
    }
protected:
    /**
     @brief deinitializer.
     */
    RuntimeCreator() = default;
};

/**
 * @brief get registered backend creator for given forward type.
 * @param type  given forward type.
 * @return backend creator pointer if registered, nullptr otherwise.
 */
MNN_PUBLIC const RuntimeCreator* MNNGetExtraRuntimeCreator(MNNForwardType type);

/**
 * @brief register backend creator for given forward type.
 * @param type given forward type.
 * @param creator registering backend creator.
 * @return true if backend creator for given forward type was not registered before, false otherwise.
 */
MNN_PUBLIC bool MNNInsertExtraRuntimeCreator(MNNForwardType type, const RuntimeCreator* creator,
                                             bool needCheck = false);

MNN_PUBLIC bool MNNCPUCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor);
} // namespace MNN

#endif /* Backend_hpp */
