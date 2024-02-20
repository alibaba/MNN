//
//  NPUBackend.hpp
//  MNN
//
//  Created by MNN on 2019/09/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUBACKEND_H
#define MNN_NPUBACKEND_H

#include <graph/attr_value.h>
#include <graph/operator_hiai_reg.h>
#include <graph/op/all_ops.h>
#include <graph/compatible/operator_reg.h>
#include <graph/graph.h>
#include <graph/model.h>
#include <graph/compatible/all_ops.h>
#include <hiai_ir_build.h>
#include <graph/buffer.h>
#include <MNN/ErrorCode.hpp>
#include <core/Backend.hpp>
#include <core/Execution.hpp>
#include "HiAiModelManagerService.h"
#include "MNN_generated.h"

#include <stdio.h>
#include <map>
#include <memory>
#include <core/TensorUtils.hpp>

#ifdef HIAI_DEBUG
#include <android/trace.h>
#include <dlfcn.h>
#endif

using namespace std;

namespace MNN {
    typedef std::vector<Tensor *> MNNTensorList;
#ifdef HIAI_DEBUG
    typedef void *(*fp_ATrace_beginSection) (const char* sectionName);
    typedef void *(*fp_ATrace_endSection) (void);
#endif
    void NHWC2NCHW(const float* source, float* dest, int b, int c, int area);

    static ge::DataType mapDataType(DataType src) {
        ge::DataType retVal = ge::DataType::DT_UNDEFINED;
        switch (src) {
            case DataType_DT_FLOAT:
                retVal = ge::DataType::DT_FLOAT;
                break;
            case DataType_DT_DOUBLE:
                retVal = ge::DataType::DT_DOUBLE;
                break;
            case DataType_DT_INT32:
                retVal = ge::DataType::DT_INT32;
                break;
            case DataType_DT_UINT8:
                retVal = ge::DataType::DT_UINT8;
                break;
            case DataType_DT_INT16:
                retVal = ge::DataType::DT_INT16;
                break;
            case DataType_DT_INT8:
                retVal = ge::DataType::DT_INT8;
                break;
            case DataType_DT_INT64:
                retVal = ge::DataType::DT_INT64;
                break;
            case DataType_DT_VARIANT:
                retVal = ge::DataType::DT_FLOAT;
                break;
            default:
                MNN_ASSERT(false);
                printf("cast Datatype : %d \n", src);
                break;
        }
        return retVal;
    }
    inline std::vector<int64_t> tensorShapeFormat(const Tensor *input, const Tensor *broadCastInput=nullptr) {
        auto dimSize = input->buffer().dimensions;
        if(broadCastInput != nullptr) {
            dimSize = dimSize > broadCastInput->buffer().dimensions ? dimSize: broadCastInput->buffer().dimensions;
        }
        //MNN_PRINT("tensorShapeFormat dimSize = %d\n",dimSize);
        vector<int> dims(8,1);
        int j = dimSize-1;
        for (int i = input->buffer().dimensions-1; i >= 0; i--)
        {
            dims[j] = input->buffer().dim[i].extent;
            j--;
        }
        //MNN_PRINT("tensorShapeFormat dims = %d %d %d %d\n",dims[0], dims[1], dims[2], dims[3]);
        int iN = dims[0];
        int iC = dims[1];
        int iH = dims[2];
        int iW = dims[3];

        if (TensorUtils::getDescribe(input)->dimensionFormat == MNN::MNN_DATA_FORMAT_NCHW ||
            TensorUtils::getDescribe(input)->dimensionFormat == MNN::MNN_DATA_FORMAT_NC4HW4) 
        {
            std::vector<int64_t> shape_vec{iN, iC, iH, iW};
            return shape_vec;
        }

        if (TensorUtils::getDescribe(input)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC) {
            iN = dims[0];
            iC = dims[3];
            iH = dims[1];
            iW = dims[2];
        }
        
        if (dimSize > 4) { // more than 4 dimensions put to N dimension
            for (int i = 0; i < dimSize-3; i++) {
                iN *= dims[i];
            }
            iC = dims[dimSize-3];
            iH = dims[dimSize-2];
            iW = dims[dimSize-1];
        }
        
        if (dimSize == 3) {
            iN = 1;
            iC = dims[2];
            iH = dims[0];
            iW = dims[1];
        }

        if (dimSize == 2) {
            iN = 1;
            iC = dims[1];
            iH = dims[0];
            iW = 1;
        }

        if (dimSize == 1) {
            iN = 1;
            iC = dims[0];
            iH = 1;
            iW = 1;
        }

        std::vector<int64_t> shape_vec{iN, iC, iH, iW};

        return shape_vec;
    }

    inline vector<int32_t>  convertShape(MNN_DATA_FORMAT dimFormat,vector<int32_t> dims, int defaultVal=1)
    {
        auto dimSize = dims.size();
        int iN =1;
        int iC =1;
        int iH =1;
        int iW =1;

        if (dimFormat == MNN::MNN_DATA_FORMAT_NCHW ||
            dimFormat == MNN::MNN_DATA_FORMAT_NC4HW4) 
        {
            iN = dims[0];
            iC = dims[1];
            iH = dims[2];
            iW = dims[3];
        }else if(dimFormat == MNN::MNN_DATA_FORMAT_NHWC) {
            iN = dims[0];
            iC = dims[3];
            iH = dims[1];
            iW = dims[2];
        }
        
        if (dimSize > 4) { // more than 4 dimensions put to N dimension
            for (int i = 0; i < dimSize-3; i++) {
                iN *= dims[i];
            }
            iC = dims[dimSize-3];
            iH = dims[dimSize-2];
            iW = dims[dimSize-1];
        }else if (dimSize == 3) {
            iN = defaultVal;
            iC = dims[2];
            iH = dims[0];
            iW = dims[1];
        }else if (dimSize == 2) {
            iN = defaultVal;
            iC = dims[1];
            iH = dims[0];
            iW = defaultVal;
        }else if (dimSize == 1) {
            iN = defaultVal;
            iC = dims[0];
            iH = defaultVal;
            iW = defaultVal;
        }
        std::vector<int32_t> newShape{iN, iC, iH, iW};
        return newShape;
    };

    inline vector<int32_t>  convertShapeConstValue(Tensor *input, int defaultVal=1)
    {
        auto dimSize = input->buffer().dim[0].extent;
        vector<int32_t> dims(dimSize,1);
        for(auto i=0; i<dimSize; i++) {
            dims[i] = input->host<int32_t>()[i];
        }
        return convertShape(TensorUtils::getDescribe(input)->dimensionFormat,dims, defaultVal);
    }

    inline int32_t convertMask(Tensor *input, int32_t mask, int defaultVal=0)
    {
        auto dimSize = input->buffer().dim[0].extent;
        vector<int32_t> dims(dimSize,0);
        for(auto i = 0; i < dimSize; i++) {
            dims[i] = mask & (1 << i);
        }
        auto newDims = convertShape(TensorUtils::getDescribe(input)->dimensionFormat, dims, defaultVal);
        int32_t newMask = 0;
        for(auto i = 0; i < newDims.size(); i++) {
            int bitVal = newDims[i] > 0 ? 1 : 0;
            newMask = newMask | (bitVal<<i); 
        }
        return newMask;
    }

    inline int axisFormat(const Tensor *input, int axis) {
        int axis_nchw;
        if(axis < 0){
            axis_nchw = input->buffer().dimensions + axis;
        }else{
            axis_nchw = axis;
        }

        if (TensorUtils::getDescribe(input)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC) {
            vector<int> dims2{2, 1};
            vector<int> dims3{2, 3, 1};
            vector<int> dims4{0, 2, 3, 1};
            if (input->buffer().dimensions == 1) {
                axis_nchw = 1;
            }else if(input->buffer().dimensions == 2){
                axis_nchw = dims2[axis_nchw];
            }else if(input->buffer().dimensions == 3){
                axis_nchw = dims3[axis_nchw];
            }else if(input->buffer().dimensions == 4){
                axis_nchw = dims4[axis_nchw];
            }
        }
        return axis_nchw;
    }

    class NPURuntime : public Runtime {
    public:
        NPURuntime(const Backend::Info& info);
        virtual ~NPURuntime();
        virtual CompilerType onGetCompilerType() const override;
        virtual Backend* onCreate(const BackendConfig* conf) const override;
        virtual void onGabageCollect(int level) override;
        // If buffer is not nullptr, try copy cache, else delete cache
        virtual bool onSetCache(const void* buffer, size_t size) override {
            // if (nullptr == buffer) {
            //     // Destroy cache
            //     if (nullptr != mModel) {
            //         mModel->destroy();
            //         mModel = nullptr;
            //     }
            //     mCacheBuffer = nullptr;
            //     mCacheSize = 0;
            //     return true;
            // }
            // mCacheBuffer = buffer;
            // mCacheSize = size;
            return true;
        }

        virtual std::pair<const void*, size_t> onGetCache() override {
            // if (mModel != nullptr) {
            //     return std::make_pair(mModel->data(), mModel->size());
            // }
            return std::make_pair(mCacheBuffer, mCacheSize);
        }

    private:
        Backend::Info mInfo;
        BackendConfig::PrecisionMode mPrecision;
        // mutable IHostMemory* mModel = nullptr;
        const void* mCacheBuffer = nullptr;
        size_t mCacheSize = 0;

        friend class NPUBackend;
    };

    class NPUBackend : public Backend {
    public:

        NPUBackend(const NPURuntime* runtime);
        virtual ~NPUBackend();

        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override;

        virtual void onExecuteBegin() const override;
        virtual void onExecuteEnd() const override;

        virtual Backend::MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
        virtual bool onClearBuffer() override;
        virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

        virtual void onResizeBegin() override;
        virtual ErrorCode onResizeEnd() override;

    public:

        ErrorCode bulidIRModelAndLoad();
        int process() const ;

        shared_ptr<ge::Operator> getInputOps(const Op *op, int index = 0);

        void setOutputOps(const Op *op, vector<shared_ptr<ge::Operator>>&& HIAI_op,
                          const std::vector<Tensor *> &outputs);
        void setNetworkInput(const std::vector<Tensor *> &inputs, const Op* op);

    public:

        map<int, vector<pair<shared_ptr<ge::Operator>, string>>> mGrapMap;
        map<shared_ptr<ge::Operator>, MNNTensorList> mOutGEOpMap;

        map<int, std::vector<ge::Operator>> mInputOps;

        map<int, int> mSclipMap;
        map<unsigned long, int> mInputMap;
    public:
        class Creator {
        public:
            virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                        const MNN::Op* op, Backend* backend) const = 0;
        };

        static bool addCreator(OpType t, Creator* c);

    private:

        vector<string> mModelName;

        MNNTensorList mMNNOutTensors;
        const NPURuntime* mNPURuntime;
        BackendConfig::PrecisionMode mPrecision;

        shared_ptr<hiai::IBuiltModel> builtModel;
        shared_ptr<hiai::IModelManager> modelManager;
        vector<shared_ptr<hiai::INDTensorBuffer>> inputTensors;
        vector<shared_ptr<hiai::INDTensorBuffer>> outputTensors;

#ifdef HIAI_DEBUG
        void *(*ATrace_beginSection) (const char* sectionName);
        void *(*ATrace_endSection) (void);
#endif
    };

    template <class T>
    class NPUCreatorRegister {
    public:
        NPUCreatorRegister(OpType type) {
            T *t = new T;
            NPUBackend::addCreator(type, t);
        }
        ~NPUCreatorRegister() = default;
    };

    template <typename T>
    class TypedCreator : public NPUBackend::Creator {
    public:
        virtual ~TypedCreator() = default;
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op,
                                    Backend *backend) const override {
            auto newOp = new T(backend, op, inputs, outputs);
            return newOp;
        }
    };

}

#endif //MNN_NPUBACKEND_H
