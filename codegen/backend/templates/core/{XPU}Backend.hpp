{COPYRIGHT}

#ifndef MNN_{XPU}BACKEND_Hpp
#define MNN_{XPU}BACKEND_Hpp


{EXTRA_INCLUDE_FILES}
// MNN headers
#include <MNN/ErrorCode.hpp>
#include <core/Backend.hpp>
#include <core/Execution.hpp>
#include "MNN_generated.h"

// stl libs
#include <map>


namespace MNN {

class {XPU}Runtime : public Runtime {
public:
    // 0. Constructor and Destructor
    {XPU}Runtime(const Backend::Info& info);
    virtual ~{XPU}Runtime();
    virtual CompilerType onGetCompilerType() const override;

    // 1. Backend Creation
    virtual Backend* onCreate(const BackendConfig* conf, Backend* origin) const override;
    
    // 2. gc function
    virtual void onGabageCollect(int level) override;
    
    // 3. jit compilation functions

    // 4. jit compilation/execution info cache
    virtual bool onSetCache(const void* buffer, size_t size) override;
    virtual std::pair<const void*, size_t> onGetCache() override;

private:
    ErrorCode initDevice();
    ErrorCode setupResourcePool();
    void releaseResourcePool(int level = 100); 

private:
    Backend::Info mInfo;
    BackendConfig::PowerMode mPower;
    BackendConfig::MemoryMode mMemory;
    BackendConfig::PrecisionMode mPrecision;
    {RUNTIME_PARAMS}

    // friend class declaration
    friend class {XPU}Backend;
};

// < {XPU}Backend begin
class {XPU}Backend : public Backend {
public:
    // 0. Constructor and Destructor
    {XPU}Backend(MNNForwardType type, {XPU}Runtime* rt);
    virtual ~{XPU}Backend();

    // 1. Execution Registration & Creation
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override;
    class Creator {
    public:
        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op, Backend* backend) const = 0;
    };
    static bool addCreator(OpType t, Creator* c);

    // 2. Pipeline Functions
    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;
    virtual void onResizeBegin() override; // If inherit default, do nothing
    virtual ErrorCode onResizeEnd() override;

    // 3. Buffer Management
    virtual MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
    // MNN_PUBLIC bool onAcquireBuffer(const Tensor* tensor, StorageType storageType);
    // MNN_PUBLIC bool onReleaseBuffer(const Tensor* tensor, StorageType storageType);
    virtual bool onClearBuffer() override;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

private:
    void initParam();
    ErrorCode jitPreBuild();
    ErrorCode jitPreTuning();

private:
    const {XPU}Runtime* mRuntime;
    BackendConfig::PowerMode mPower;
    BackendConfig::MemoryMode mMemory;
    BackendConfig::PrecisionMode mPrecision;
};

// 1. Execution Creator Register
template <class T>
class {XPU}CreatorRegister {
public:
    {XPU}CreatorRegister(OpType type) {
        T *t = new T; {XPU}Backend::addCreator(type, t);
    }
    ~{XPU}CreatorRegister() = default;
};

// {XPU}Backend end>

} // MNN

#endif