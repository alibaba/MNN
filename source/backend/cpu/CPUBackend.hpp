//
//  CPUBackend.hpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUBackend_hpp
#define CPUBackend_hpp

#include <stdio.h>
#include <map>
#include <memory>
#include "core/Backend.hpp"
#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class BufferAllocator;

class CPUBackend final : public Backend {
public:
    CPUBackend(int numberThread = 4, BackendConfig::MemoryMode memory = BackendConfig::Memory_Normal,
               BackendConfig::PowerMode = BackendConfig::Power_Normal, size_t flags = 0);
    virtual ~CPUBackend();

    // Return sizeDivide, scheduleNumber aligned memory
    std::pair<int, int> multiThreadDivide(int size) const;
public:
    virtual bool onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) override;
    virtual bool onReleaseBuffer(const Tensor* nativeTensor, StorageType storageType) override;
    virtual bool onAllocateBuffer() override;
    virtual bool onClearBuffer() override;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;
    virtual std::pair<float, bool> onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                            const MNN::Op* op) override;

    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;
    
    virtual void* getAllocator(StorageType type = DYNAMIC) const override{
        if(type == STATIC){
            return (void*)mStaticAllocator.get();
        }
        return (void*)mDynamicAllocator.get();
    }
public:
    class Creator {
    public:
        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op, Backend* backend) const = 0;
    };

    static bool addCreator(OpType t, Creator* c);

    int threadNumber() const {
        return mThreadNumber;
    }

    BufferAllocator* getBufferAllocator() const {
        return static_cast<BufferAllocator*>(this->getAllocator());
    }

    BackendConfig::MemoryMode memoryMode() const {
        return mMemory;
    }
    BackendConfig::PowerMode powerMode() const {
        return mPower;
    }
#ifdef MNN_USE_THREAD_POOL
    inline int taskIndex() const {return mTaskIndex;}
#endif

private:
    std::unique_ptr<BufferAllocator> mStaticAllocator;
    std::unique_ptr<BufferAllocator> mDynamicAllocator;
    int mThreadNumber;
#ifdef MNN_USE_THREAD_POOL
    int mTaskIndex;
#endif
    const BackendConfig::MemoryMode mMemory;
    const BackendConfig::PowerMode mPower;
    bool mCheckNAN = false;
    float mFlops = 0.0f;
};

#ifdef MNN_CODEGEN_REGISTER
#define REGISTER_CPU_OP_CREATOR(name, opType)     \
    void ___##name##__##opType##__() {            \
        CPUBackend::addCreator(opType, new name); \
    }
#else

template <class T>
class CPUCreatorRegister {
public:
    CPUCreatorRegister(OpType type) {
        CPUBackend::addCreator(type, new T);
    }
};

#define REGISTER_CPU_OP_CREATOR(name, opType) static CPUCreatorRegister<name> _Create##opType(opType)
#endif

} // namespace MNN

#endif /* CPUBackend_hpp */
