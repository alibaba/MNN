//
//  TRTBackend.hpp
//  MNN
//
//  Created by MNN on 2019/09/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTBackend_H
#define MNN_TRTBackend_H

#include <MNN/ErrorCode.hpp>
#include <core/Backend.hpp>
#include <core/Execution.hpp>

#include "MNN_generated.h"

#include <stdio.h>
#include <core/TensorUtils.hpp>
#include <map>
#include <memory>
#include "TRTPlugin.hpp"
#include "TRTType.hpp"
#include "core/ConvolutionCommon.hpp"
#include "cuda_runtime.h"

using namespace std;
using namespace nvinfer1;

namespace MNN {

class TRTRuntime : public Runtime {
public:
    TRTRuntime(const Backend::Info& info);
    virtual ~TRTRuntime();

    virtual Backend* onCreate(const BackendConfig* config) const override;
    virtual void onGabageCollect(int level) override;
    // If buffer is not nullptr, try copy cache, else delete cache
    virtual bool onSetCache(const void* buffer, size_t size) override {
        if (nullptr == buffer) {
            // Destroy cache
            if (nullptr != mModel) {
                mModel->destroy();
                mModel = nullptr;
            }
            mCacheBuffer = nullptr;
            mCacheSize = 0;
            return true;
        }
        mCacheBuffer = buffer;
        mCacheSize = size;
        return true;
    }
    virtual CompilerType onGetCompilerType() const override {
        return Compiler_Geometry;
    }

    virtual std::pair<const void*, size_t> onGetCache() override {
        if (mModel != nullptr) {
            return std::make_pair(mModel->data(), mModel->size());
        }
        return std::make_pair(mCacheBuffer, mCacheSize);
    }

private:
    Backend::Info mInfo;
    BackendConfig::PrecisionMode mPrecision;
    mutable IHostMemory* mModel = nullptr;
    const void* mCacheBuffer = nullptr;
    size_t mCacheSize = 0;

    friend class TRTBackend;
};

class TRTBackend : public Backend {
public:
    TRTBackend(const TRTRuntime* runtime);
    virtual ~TRTBackend();

    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;

    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;

    virtual Backend::MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

    virtual void onResizeBegin() override;
    virtual ErrorCode onResizeEnd() override;

    class Creator {
    public:
        virtual ~Creator()                                                     = default;
        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& output,
                                    const MNN::Op* op, Backend* backend) const = 0;
    };

    static bool addCreator(OpType t, Creator* c);

    INetworkDefinition* getNetwork();
    void cudaErrorCheck(string tag = "") const;

    ITensor* getTensorOps(const Tensor* inputs);
    void setTensorOps(const std::vector<Tensor*>& outputs, vector<ITensor*>&& TRT_op);

    void init();
    void unInit();

    void pushCache(std::shared_ptr<ConvolutionCommon::Int8Common> cache) const {
        mCache.push_back(cache);
    }
    void pushReleaseLayer(nvinfer1::IPluginExt* layer) const {
        mEraseLayers.push_back(layer);
    }

private:
    mutable std::vector<std::shared_ptr<ConvolutionCommon::Int8Common>> mCache; // should be deleted after init
    mutable std::map<const Tensor*, std::pair<ITensor*, int>> mTensorMaps;
    mutable std::map<const Tensor*, std::pair<std::string, void*>> mInputs;
    mutable std::map<const Tensor*, std::pair<std::string, void*>> mOutputs;
    mutable std::vector<nvinfer1::IPluginExt*> mEraseLayers;

    IRuntime* mRuntime{nullptr};
    ICudaEngine* mEngine{nullptr};
    IExecutionContext* mContext{nullptr};
    INetworkDefinition* mNetwork{nullptr};
    IBuilder* mBuilder{nullptr};
    std::vector<void*> mInOutbuffers;
    BackendConfig::PrecisionMode mPrecision;
    const TRTRuntime* mTRTRuntime;
};

template <class T>
class TRTCreatorRegister {
public:
    TRTCreatorRegister(OpType type) {
        T* t = new T;
        TRTBackend::addCreator(type, t);
    }
    ~TRTCreatorRegister() = default;
};

template <typename T>
class TypedCreator : public TRTBackend::Creator {
public:
    virtual ~TypedCreator() = default;
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new T(backend, op, inputs, outputs);
    }
};

} // namespace MNN

#endif // MNN_TRTBackend_H
