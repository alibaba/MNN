//
//  MusaBackend.hpp
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#ifndef MusaBackend_hpp
#define MusaBackend_hpp

#include <set>
#include <vector>
#include <MNN/ErrorCode.hpp>
#include "MNN_generated.h"
#include "backend/musa/core/runtime/MusaRuntime.hpp"
#include "core/Backend.hpp"
#include "core/Macro.h"
#include "core/ConvolutionCommon.hpp"
#include "core/BufferAllocator.hpp"
#include "backend/cpu/CPUResizeCache.hpp"

#define MNN_USER_SET_DEVICE
#include "MNN/MNNSharedContext.h"

// Pack numbers for GPU operations
#ifndef PACK_NUMBER
#define PACK_NUMBER 4
#endif
#ifndef INT8_PACK_NUMBER
#define INT8_PACK_NUMBER 16
#endif

namespace MNN {
namespace MUSA {

class MNN_PUBLIC MusaRuntimeWrapper : public Runtime {
public:
    MusaRuntimeWrapper(BackendConfig::PrecisionMode precision, BackendConfig::PowerMode power, BackendConfig::MemoryMode memory, int deviceId = 0);
    virtual ~MusaRuntimeWrapper();
    virtual Backend *onCreate(const BackendConfig* config, Backend* origin) const override;
    virtual void onGabageCollect(int level) override;
    bool isCreateError() const {
        return mIsCreateError;
    }
    virtual CompilerType onGetCompilerType() const override {
        return Compiler_Loop;
    }
    virtual float onGetMemoryInMB() override;
    virtual std::pair<const void*, size_t> onGetCache() override;
    virtual bool onSetCache(const void* buffer, size_t size) override;

private:
    std::shared_ptr<EagerBufferAllocator> mBufferPool;
    std::shared_ptr<MusaRuntime> mMusaRuntime;
    bool mIsCreateError{false};
    BackendConfig::PrecisionMode mDefaultPrecision;
    BackendConfig::MemoryMode mDefaultMemory;
};

class MusaBackend : public Backend {
public:
    MusaBackend(std::shared_ptr<BufferAllocator> st, std::shared_ptr<MusaRuntime> rt, int precisionLevel, BackendConfig::MemoryMode memoryLevel);
    ~MusaBackend();

    MusaRuntime *getMusaRuntime();
    virtual const Runtime* getRuntime() override;
    virtual Backend::MemObj* onAcquire(const Tensor *nativeTensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;

    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op) override;
    virtual void onResizeBegin() override;
    virtual ErrorCode onResizeEnd() override;

    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;

    virtual void onCopyBuffer(const Tensor *srcTensor, const Tensor *dstTensor) const override;
    virtual int onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) override;

    class Creator {
    public:
        virtual ~Creator()                                                     = default;
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &output,
                                    const MNN::Op *op, Backend *backend) const = 0;
    };

    static bool addCreator(OpType t, Creator *c);
    static DataType getDataType(const Tensor* tensor);

    BufferAllocator *getBufferPool() const {
        return mBufferPool.get();
    }
    BufferAllocator *getStaticBufferPool() const {
        return mStaticBufferPool.get();
    }
    static size_t realSize(const Tensor *tensor);
    int getBytes(const Tensor* tensor) const;
    CPUResizeCache* getCache();
    bool useFp16() const;
    int getPrecision() const;
    BackendConfig::MemoryMode getMemoryMode() const;

private:
    std::shared_ptr<BufferAllocator> mBufferPool;
    std::shared_ptr<BufferAllocator> mStaticBufferPool;
    std::shared_ptr<MusaRuntime> mMusaRuntime;
    CPUResizeCache mCache;
    bool mUseFp16AsFp32 = false;
    int mPrecision = 0;
    BackendConfig::MemoryMode mMemory;
};

template <class T>
class MusaCreatorRegister {
public:
    MusaCreatorRegister(OpType type) {
        T *t = new T;
        MusaBackend::addCreator(type, t);
    }
    ~MusaCreatorRegister() = default;
};

/** execution cast wrapper. insert tensor cast dynamic. */
class CastWrapExecution : public Execution {
public:
    CastWrapExecution(Backend* backend, DataType runT)
                    : Execution(backend), mRunType(runT) {}
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
private:
    DataType mRunType;
};

template <typename T>
class TypedCreator : public MusaBackend::Creator {
public:
    virtual ~TypedCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new T(inputs, op, backend);
    }
};

} // namespace MUSA
} // namespace MNN
#endif /* MusaBackend_hpp */
