//
//  ConvFpAIntBExecution.hpp
//  MNN
//
//  Created by MNN on 2024/03/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef ConvFpAIntBExecution_hpp
#define ConvFpAIntBExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "../MNNCUDADefine.hpp"
#include "../MNNCUDAFunction.cuh"
#include "../cutlass_common/CutlassConvCommonExecution.hpp"

namespace MNN {
namespace CUDA {

class ConvFpAIntBExecution : public CutlassConvCommonExecution {
public:
    struct Resource {
        Resource(Backend* bn, const MNN::Op* op);
        ~ Resource();
        void* mFilter;
        void* mScale;
        void* mOffset;
        void* mBias;
        int mQuanC;
        std::shared_ptr<Tensor> weightTensor;
        std::shared_ptr<Tensor> scaleTensor;
        std::shared_ptr<Tensor> offsetTensor;
        std::shared_ptr<Tensor> biasTensor;
        Backend* mBackend = nullptr;
        bool mIsWeightInt4 = false;
        
        std::shared_ptr<Tensor> mSumBQTensor;
        void* mSumBQ = nullptr;

        // R6: Pre-computed GEMV params: float2 array [oc * num_qg]
        // .x = scale (float32), .y = adj_offset = offset - 8*scale (float32)
        // Eliminates scattered offset loads and runtime adj_off computation
        float2* mGemvParams = nullptr;  // allocated via cudaMalloc
        int mNumQg = 0;  // number of quantization groups per OC
    };
    static bool isValid(const Convolution2D* conv, Backend* backend);
    ConvFpAIntBExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res);
    virtual ~ConvFpAIntBExecution();
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<Resource> mResource;

    std::shared_ptr<Tensor> mDequantFilterTensor;
    void* mDequantFilter = nullptr;

    // When true, STATIC dequant buffer allocation failed (OOM for large models).
    // Dequantization will be done just-in-time in onExecute using DYNAMIC buffer.
    bool mNeedRuntimeDequant = false;
    bool mDequantIsStatic = false;  // Track whether dequant buffer is STATIC or DYNAMIC

};

} // namespace CUDA
} // namespace MNN

#endif /* ConvFpAIntBExecution */