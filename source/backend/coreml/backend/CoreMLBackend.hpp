//
//  CoreMLBackend.hpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLBACKEND_H
#define MNN_COREMLBACKEND_H

#include <MNN/ErrorCode.hpp>
#include <core/Backend.hpp>
#include <core/Execution.hpp>
#include <core/TensorUtils.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <functional>
#include "CoreMLExecutorWrapper.h"
#include "CoreMLHelper.hpp"
#include "MNN_generated.h"
#include "core/BufferAllocator.hpp"
#include "core/AutoStorage.h"

namespace MNN {
struct CoreMLLayerState {
    std::mutex mutex;
    std::shared_ptr<CoreMLExecutorWrapper> session;
};
struct CoreMLWeightSource {
    std::function<void(int16_t*)> decode;
    size_t elements = 0, storageBytes = 0;
};
struct CoreMLWeights {
    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    std::map<std::string, std::shared_ptr<CoreMLWeightSource>> streamed;
};
struct CoreMLProgramGroup {
    std::shared_ptr<CoreMLExecutorWrapper> session;
    std::vector<std::pair<const Tensor*, std::string>> inputs, outputs, staging;
    std::vector<std::shared_ptr<Tensor>> storage;
};
struct CoreMLProgramContext {
    MlModelBuilder builder{9};
    bool valid = true;
    bool flatIO = false;
    std::shared_ptr<CoreMLProgramGroup> group;
    bool lastInGroup = false;
    std::vector<std::shared_ptr<Tensor>> layoutViews;
    std::map<const Tensor*, std::string> names;
    std::set<const Tensor*> outputs;
    std::set<const Tensor*> constants;
    std::vector<std::pair<const Tensor*, std::string>> bindings;
    std::set<std::string> boundNames;
    std::shared_ptr<CoreMLWeights> weights = std::make_shared<CoreMLWeights>();
    std::map<std::string, std::shared_ptr<Tensor>> weightViews;
    std::shared_ptr<CoreMLExecutorWrapper> program, session;
    std::shared_ptr<CoreMLLayerState> layerState = std::make_shared<CoreMLLayerState>();
};

class CoreMLRuntime : public Runtime {
public:
    CoreMLRuntime(const Backend::Info& info);
    ~CoreMLRuntime() override;
    CompilerType onGetCompilerType() const override;
    Backend* onCreate(const BackendConfig* conf, Backend* origin) const override;
    void onGabageCollect(int level) override;
    std::shared_ptr<CoreMLExecutorWrapper> getProgram(CoreML__Specification__Model* model,
                                                      const std::string& weightsPath = "",
                                                      const std::shared_ptr<std::vector<uint8_t>>& weights = nullptr,
                                                      bool forceFileCompilation = false) const;
    size_t compilationCount() const;
    size_t cacheHitCount() const;
    std::shared_ptr<CoreMLWeights> findWeights(const std::string& key) const;
    void cacheWeights(const std::string& key, const std::shared_ptr<CoreMLWeights>& weights) const;
    bool lowMemory() const { return mMemory == BackendConfig::Memory_Low; }
    std::unique_lock<std::mutex> lockWeights() const { return std::unique_lock<std::mutex>(mWeightExecutionMutex); }
    bool prepareWeights(const CoreMLProgramContext& context) const;

private:
    BackendConfig::PrecisionMode mPrecision;
    BackendConfig::MemoryMode mMemory;
    struct CacheEntry {
        std::shared_ptr<CoreMLExecutorWrapper> program;
        size_t lastUse;
    };
    mutable std::mutex mCacheMutex;
    mutable std::map<std::string, CacheEntry> mPrograms;
    mutable size_t mCompilations = 0, mCacheHits = 0, mClock = 0, mProgramLoads = 0;
    mutable std::mutex mWeightMutex;
    mutable std::map<std::string, std::weak_ptr<CoreMLWeights>> mWeights;
    mutable std::mutex mWeightExecutionMutex;
    mutable AutoStorage<int16_t> mWeightScratch;
    mutable size_t mWeightHits = 0, mWeightLoads = 0, mWeightBytes = 0, mQuantizedBytes = 0;
};

class CoreMLBackend : public Backend {
public:
    explicit CoreMLBackend(const CoreMLRuntime* runtime);
    ~CoreMLBackend() override = default;
    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op) override;
    void* onMapTensor(Tensor::MapType, Tensor::DimensionType, const Tensor*) override { return nullptr; }
    bool onUnmapTensor(Tensor::MapType, Tensor::DimensionType, const Tensor*, void*) override { return false; }
    void onExecuteBegin() const override {}
    void onExecuteEnd() const override {}
    MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
    bool onClearBuffer() override;
    void onCopyBuffer(const Tensor* src, const Tensor* dst) const override;
    void onResizeBegin() override { mPending.clear(); }
    ErrorCode onResizeEnd() override;
    ErrorCode onResizeEnd(const std::vector<Command*>& commands) override;
    void recordProgram(Execution* execution, CoreMLProgramContext* context, const std::vector<Tensor*>& outputs) {
        mPending[execution] = {context, outputs};
    }
    const Runtime* getRuntime() override { return mRuntime; }

    void setProgramContext(CoreMLProgramContext* context) { mContext = context; }
    ErrorCode buildCallbackProgram(Execution* execution, CoreMLProgramContext& context,
                                   const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    ErrorCode buildProgram(CoreMLProgramContext& context, const std::vector<Tensor*>& outputs);
    std::string getTensorName(const Tensor* tensor);
    void useFlatIO() { mContext->flatIO = true; }
    std::vector<int> tensorShape(const Tensor* tensor) const {
        return mContext->flatIO ? std::vector<int>{static_cast<int>(TensorUtils::getRawSize(tensor))} : tensor->shape();
    }
    void replaceTensorWith(const Tensor* src, const Tensor* replace);
    std::string
    addState(const std::string& name, const std::vector<int>& shape,
             CoreML__Specification__MILSpec__DataType dtype = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT16);
    void addOperation(std::unique_ptr<MilOperationBuilder> builder);
    std::string buildBlobConst(const void* data, size_t size, CoreML__Specification__MILSpec__DataType dtype,
                               const std::vector<int>& shape, std::string name = "");
    template <typename T>
    std::string buildArray(const std::vector<T> data, std::string name) {
        std::unique_ptr<MilOperationBuilder> op(new MilOperationBuilder("const"));
        op->setName(name);
        op->addOutput(name, MilTypeTraits<T>::dataType, {(int64_t)data.size()});
        op->addAttribute("val", MilValueBuilder::createMilValueBuilder(data));
        addOperation(std::move(op));
        return name;
    }
    template <typename T>
    std::string buildScalar(const T data, std::string name) {
        std::unique_ptr<MilOperationBuilder> op(new MilOperationBuilder("const"));
        op->setName(name);
        op->addOutput(name, MilTypeTraits<T>::dataType, {});
        op->addAttribute("val", MilValueBuilder::createMilValueBuilder(data));
        addOperation(std::move(op));
        return name;
    }
    std::string buildScalarFloat32(float data, const std::string& name) {
        auto value = MilValueBuilder::createMilValueBuilder(data);
        value->type = MilValueBuilder::Type::SCALAR_FLOAT32;
        std::unique_ptr<MilOperationBuilder> op(new MilOperationBuilder("const"));
        op->setName(name);
        op->addOutput(name, CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32, {});
        op->addAttribute("val", std::move(value));
        addOperation(std::move(op));
        return name;
    }
    std::string buildShape(const std::vector<int> data, std::string name) {
        std::unique_ptr<MilOperationBuilder> op(new MilOperationBuilder("const"));
        op->setName(name);
        op->addOutput(name, CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32, {(int64_t)data.size()});
        op->addAttribute("val", MilValueBuilder::createMilShapeBuilder(data));
        addOperation(std::move(op));
        return name;
    }
    // Bind an execution-owned input (for example the KV write position or attention mask).
    std::string addInput(const Tensor* tensor, const std::string& name);
    std::string existingWeight(const std::string& name) const {
        return mContext->weights->tensors.count(name) ? name : "";
    }
    const Tensor* weightTensor(const std::string& name) const {
        auto found = mContext->weights->tensors.find(name);
        return found == mContext->weights->tensors.end() ? nullptr : found->second.get();
    }
    void releaseWeight(const std::string& name) { mContext->weights->tensors.erase(name); }
    bool restoreWeights(const std::string& key);
    void cacheWeights(const std::string& key);
    Tensor* allocateWeight(const std::string& name, const std::vector<int>& shape,
                           std::shared_ptr<CoreMLWeightSource> source = nullptr);
    bool lowMemory() const { return mRuntime->lowMemory(); }
    CoreML__Specification__MILSpec__DataType typeConvert(const halide_type_t& type) const;
    int getBytes(const halide_type_t& type) const;
    class Creator {
    public:
        virtual ~Creator() = default;
        virtual Execution* onCreate(const std::vector<Tensor*>&, const std::vector<Tensor*>&, const Op*,
                                    Backend*) const = 0;
    };
    static bool addCreator(OpType type, Creator* creator);

private:
    struct PendingProgram {
        CoreMLProgramContext* context;
        std::vector<Tensor*> outputs;
    };
    ErrorCode buildProgramGroups(const std::vector<Command*>& commands,
                                 const std::map<Execution*, PendingProgram>& pending, bool forceFileCompilation = false);
    std::map<Execution*, PendingProgram> mPending;
    CoreMLProgramContext* mContext = nullptr;
    const CoreMLRuntime* mRuntime;
    std::shared_ptr<EagerBufferAllocator> mBuffers;
};

template <typename T>
class TypedCreator : public CoreMLBackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op,
                        Backend* backend) const override {
        return new T(backend, op, inputs, outputs);
    }
};
#define REGISTER_COREML_OP_CREATOR(name, opType)     \
    void ___##name##__##opType##__() {               \
        static TypedCreator<name> creator;           \
        CoreMLBackend::addCreator(opType, &creator); \
    }
} // namespace MNN
#endif
