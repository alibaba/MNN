//
//  NNAPIBackend.hpp
//  MNN
//
//  Created by MNN on 2022/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NNAPIBACKEND_H
#define MNN_NNAPIBACKEND_H

#include <stdio.h>
#include <map>
#include <memory>
#include <MNN/ErrorCode.hpp>
#include <core/Backend.hpp>
#include <core/Execution.hpp>
#include <core/TensorUtils.hpp>
#include "MNN_generated.h"
#include "NNAPIDefine.hpp"
#include "NNAPISymbol.hpp"

namespace MNN {
    static void dumpTensor(const Tensor* t) {
        printf("Tensor Dump %p : {\n", t);
        t->printShape();
        const char* dtype = "";
        if (t->getType() == halide_type_of<float>()) {
            dtype = "float32";
        } else if (t->getType() == halide_type_of<int8_t>()) {
            dtype = "int8_t";
        }
        printf("\t**Tensor dtype**: %s,\n", dtype);
        const char* format = "";
        if (TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            format = "NC4HW4";
        } else if (TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NCHW) {
            format = "NCHW";
        } else if (TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
            format = "NHWC";
        }
        printf("\t**Tensor format**: %s,\n", format);
        if (TensorUtils::getDescribe(t)->quantAttr.get()) {
            printf("\t**Tensor quant**: %f, %f\n", TensorUtils::getDescribe(t)->quantAttr->scale, TensorUtils::getDescribe(t)->quantAttr->zero);
        }
        printf("}\n");
    }
    class NNAPIRuntime : public Runtime {
    public:
        NNAPIRuntime(const Backend::Info& info);
        virtual ~NNAPIRuntime();
        virtual CompilerType onGetCompilerType() const override;
        virtual Backend* onCreate(const BackendConfig* conf) const override;
        virtual void onGabageCollect(int level) override;
        virtual std::pair<const void*, size_t> onGetCache() override {
            return std::make_pair(mCacheBuffer, mCacheSize);
        }

    private:
        Backend::Info mInfo;
        BackendConfig::PrecisionMode mPrecision;
        const void* mCacheBuffer = nullptr;
        size_t mCacheSize = 0;

        friend class NNAPIBackend;
    };

    class NNAPIBackend : public Backend {
    public:

        NNAPIBackend(const NNAPIRuntime* runtime);
        virtual ~NNAPIBackend();

        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override;

        virtual void onExecuteBegin() const override;
        virtual void onExecuteEnd() const override;

        virtual Backend::MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
        virtual bool onClearBuffer() override;
        virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

        virtual void onResizeBegin() override;
        virtual ErrorCode onResizeEnd() override;

    public:
        class Creator {
        public:
            virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                        const MNN::Op* op, Backend* backend) const = 0;
        };
        static bool addCreator(OpType t, Creator* c);
        // NNAPI functions
        bool NCHW() const { return mNCHW; }
        int bytes() const {
#ifdef MNN_USE_ARMV82
            return mPrecision >= BackendConfig::PrecisionMode::Precision_Low ? 2 : 4;
#else
            return 4;
#endif
        }
        template <typename T> void dimsFormat(std::vector<T>& dims, MNN_DATA_FORMAT format) {
            if (dims.size() == 4) {
                if (format != MNN_DATA_FORMAT_NHWC && !mNCHW) {
                    T n = dims[0], c = dims[1], h = dims[2], w = dims[3];
                    dims[0] = n;
                    dims[1] = h;
                    dims[2] = w;
                    dims[3] = c;
                }
            }
        }
        uint32_t getTensorIdx(const Tensor* t, bool dequant = false);
        uint32_t buildScalar(int scalar);
        uint32_t buildScalar(bool scalar);
        uint32_t buildScalar(float scalar);
        uint32_t buildOperand(const void* data, size_t size, OperandCode code, std::vector<uint32_t> dims = {}, const float* scales = nullptr, int zero = 0);
        ErrorCode buildOperation(int op, const std::vector<uint32_t> &inputs, const std::vector<uint32_t> &outputs, const char* name = nullptr);
        ErrorCode buildQuantOperation(const Tensor* src, const Tensor* dst);
        ErrorCode replaceTensorWith(const Tensor* src, const Tensor* replace);
        uint32_t buildDequantOperand(const Tensor* t);
        ErrorCode buildModel();
        void invokeModel() const;
    private:
        bool mNCHW = false;
        std::vector<std::string> mModelName;
        const NNAPIRuntime* mNPURuntime;
        BackendConfig::PrecisionMode mPrecision;
        std::vector<std::unique_ptr<Tensor>> mInputContentTensors, mOutputContentTensors;
        std::vector<const Tensor*> mInputTensors, mOutputTensors;
        // tensor idx map
        std::map<const Tensor*, uint32_t> mTensorIdxMap, mInputIdxMap, mOutputIdxMap, mDequantIdxMap;
        std::map<uint32_t, const Tensor*> mDequantMap;
        std::map<uint32_t, uint32_t> mQuantCacheMap;
        uint32_t mTensorIdx = 0;
        std::vector<const char*> mOpNames;
        // scalar idx map
        std::map<int, uint32_t> mScalarIntMap;
        std::map<bool, uint32_t> mScalarBoolMap;
        std::map<float, uint32_t> mScalarFloatMap;
        // fp16 buffer
        std::vector<std::unique_ptr<int16_t[]>> mHalfBuffer;
        // NNAPI resource
        struct NNAPIDevice {
            ANeuralNetworksDevice* device;
            const char* name;
            int32_t type;
        };
        std::vector<NNAPIDevice> mNNAPIDevices;
        ANeuralNetworksModel *mNNAPIModel = nullptr;
        ANeuralNetworksCompilation *mNNAPICompilation = nullptr;
        ANeuralNetworksBurst* mNNAPIBurst = NULL;
    };

    template <class T>
    class NNAPICreatorRegister {
    public:
        NNAPICreatorRegister(OpType type) {
            T *t = new T;
            NNAPIBackend::addCreator(type, t);
        }
        ~NNAPICreatorRegister() = default;
    };

    template <typename T>
    class TypedCreator : public NNAPIBackend::Creator {
    public:
        virtual ~TypedCreator() = default;
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op,
                                    Backend *backend) const override {
            auto newOp = new T(backend, op, inputs, outputs);
            return newOp;
        }
    };

#define REGISTER_NNAPI_OP_CREATOR(name, opType)     \
    void ___##name##__##opType##__() {            \
        static TypedCreator<name> _temp;\
        NNAPIBackend::addCreator(opType, &_temp); \
    }

}

#endif //MNN_NNAPIBACKEND_H
