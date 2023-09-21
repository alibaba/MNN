//
//  CoreMLBackend.hpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLBACKEND_H
#define MNN_COREMLBACKEND_H

#include <stdio.h>
#include <map>
#include <memory>
#include <MNN/ErrorCode.hpp>
#include <core/Backend.hpp>
#include <core/Execution.hpp>
#include <core/TensorUtils.hpp>
#include "MNN_generated.h"
#include "Model.pb-c.h"
#include "CoreMLExecutorWrapper.h"

namespace MNN {
    class CoreMLRuntime : public Runtime {
    public:
        CoreMLRuntime(const Backend::Info& info);
        virtual ~CoreMLRuntime();
        virtual CompilerType onGetCompilerType() const override;
        virtual Backend* onCreate(const BackendConfig* conf) const override;
        virtual void onGabageCollect(int level) override;
        virtual std::pair<const void*, size_t> onGetCache() override {
            return std::make_pair(mCacheBuffer, mCacheSize);
        }

    private:
        Backend::Info mInfo;
        BackendConfig::PrecisionMode mPrecision;
        // mutable IHostMemory* mModel = nullptr;
        const void* mCacheBuffer = nullptr;
        size_t mCacheSize = 0;

        friend class CoreMLBackend;
    };

    class CoreMLBackend : public Backend {
    public:

        CoreMLBackend(const CoreMLRuntime* runtime);
        virtual ~CoreMLBackend();

        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override;

        virtual void onExecuteBegin() const override;
        virtual void onExecuteEnd() const override;

        virtual Backend::MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
        virtual bool onClearBuffer() override;
        virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

        virtual void onResizeBegin() override;
        virtual ErrorCode onResizeEnd() override;

    public:
        // TODO: using memory pool instead static factory
        template <class T> class PtrContainer {
            std::vector<T*> ptr_container;
            std::vector<T*> array_container;
        public:
            ~PtrContainer() {
                for (auto t : ptr_container) {
                    delete t;
                }
                for (auto t : array_container) {
                    delete [] t;
                }
            }
            void insert(T* t) {
                ptr_container.push_back(t);
            }
            void add(T* t) {
                array_container.push_back(t);
            }
        };
        // create C struct pointer
        template <class T> T* create() {
            static PtrContainer<T> con;
            auto t = new T;
            con.insert(t);
            return t;
        }
        template <class T> T* create(size_t size) {
            static PtrContainer<T> con;
            auto t = new T[size];
            con.add(t);
            return t;
        }
        std::string getTensorName(const Tensor* t);
        void addLayer(CoreML__Specification__NeuralNetworkLayer* layer);
        ErrorCode buildModel();
        void invokeModel() const;
        void setIO(CoreML__Specification__FeatureDescription** describe, const Tensor* t);
        void setLayerName(CoreML__Specification__NeuralNetworkLayer* layer, std::string&& name);
        void setLayerInputs(CoreML__Specification__NeuralNetworkLayer* layer, std::vector<std::string>&& inputs);
        void setLayerOutputs(CoreML__Specification__NeuralNetworkLayer* layer, std::vector<std::string>&& outputs);
        void copyName(char** ptr, std::string&& name);
        int getInOutTensorInfo(std::string modelName);

        class Creator {
        public:
            virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                        const MNN::Op* op, Backend* backend) const = 0;
        };

        static bool addCreator(OpType t, Creator* c);
    private:
        std::unique_ptr<_CoreML__Specification__Model> mCoreMLModel_;
        std::vector<CoreML__Specification__NeuralNetworkLayer*> mCoreMLLayerPtrs;

        std::map<const Tensor*, int> mTensorIdxMap, mInputIdxMap, mOutputIdxMap;
        std::vector<const Tensor*> mInputTensors;
        std::vector<std::string> mModelName;
        std::vector<std::unique_ptr<float>> mInputData, mOutputData;
        const CoreMLRuntime* mNPURuntime;
        BackendConfig::PrecisionMode mPrecision;
        std::unique_ptr<CoreMLExecutorWrapper> mCoreMLExecutor;
    };

    template <class T>
    class CoreMLCreatorRegister {
    public:
        CoreMLCreatorRegister(OpType type) {
            T *t = new T;
            CoreMLBackend::addCreator(type, t);
        }
        ~CoreMLCreatorRegister() = default;
    };

    template <typename T>
    class TypedCreator : public CoreMLBackend::Creator {
    public:
        virtual ~TypedCreator() = default;
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op,
                                    Backend *backend) const override {
            auto newOp = new T(backend, op, inputs, outputs);
            return newOp;
        }
    };

#define REGISTER_COREML_OP_CREATOR(name, opType)     \
    void ___##name##__##opType##__() {            \
        static TypedCreator<name> _temp;\
        CoreMLBackend::addCreator(opType, &_temp); \
    }

}

#endif //MNN_COREMLBACKEND_H
