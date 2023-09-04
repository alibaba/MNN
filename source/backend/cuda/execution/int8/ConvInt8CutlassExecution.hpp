//
//  ConvInt8CutlassExecution.hpp
//  MNN
//
//  Created by MNN on 2023/01/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_QUANT

#ifndef ConvInt8CutlassExecution_hpp
#define ConvInt8CutlassExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "CutlassGemmInt8Param.hpp"
#include "../MNNCUDADefine.hpp"
#include "../MNNCUDAFunction.cuh"

namespace MNN {
namespace CUDA {

typedef enum {
    GEMM_SIZE_NORMAL = 0,
    GEMM_SIZE_LITTLE = 1,
    GEMM_SIZE_LARGE  = 2
} GemmSizeLevel;

class ConvInt8CutlassExecution : public Execution {
public:
    struct Resource {
        Resource(Backend* bn, const MNN::Op* op);
        ~ Resource();
        void* mWeightInt8Ptr;
        void* mBiasInt32Ptr;
        void* mScaleFloatPtr;
        std::shared_ptr<Tensor> mWeightInt8Tensor;
        std::shared_ptr<Tensor> mBiasInt32Tensor;
        std::shared_ptr<Tensor> mScaleFloatTensor;

        int32_t* mBiasInt32Vec;
        float* mScaleFloatVec;
        Backend* mBackend = nullptr;

        // relu or relu6
        int mActivationType;
        int mActBits;

        int32_t mInputZeroPoint;
        int32_t mOutputZeroPoint;
        int8_t mClampMin;
        int8_t mClampMax;
        float mInputScale;
        float mOutputScale;
        int mOutputChannelPack;
        std::vector<int> mInt8WeightKernelSum;
        bool mUseConvQuan = true;
        void updateInputOutputScale(std::vector<float> inputQuantInfo, std::vector<float> outputQuantInfo);
    };
    ConvInt8CutlassExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res);
    virtual ~ConvInt8CutlassExecution();
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    ErrorCode callCutlassGemmInt8TensorCore(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs);
    ErrorCode callCutlassGemmInt8TensorCore16832(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs);
private:
    std::shared_ptr<Resource> mResource;

    const Op* mOp = nullptr;
    CutlassGemmInfo mGemmInfo;

    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    MemChunk mGpuIm2ColParam;

    void* mIm2ColBuffer;

    bool mIsConv1x1S1D1P0 = false;
    bool mNeedIm2Col = true;
    MemChunk mGpuKernelParam;
    bool mIsBlock = false;
    int mBlockNum = 1;

    GemmInt8Tensor_Clamp_AlignTensor_Little mGemmInt8ClampLittle;
    GemmInt8Tensor_Clamp_AlignTensor_Normal mGemmInt8ClampNormal;
    GemmInt8Tensor_Clamp_AlignTensor_Large  mGemmInt8ClampLarge;

    GemmInt8Tensor_Clamp_AlignTensor_Normal_Sm80 mGemmInt8ClampNormalSm80;
    
    GemmSizeLevel mGemmShapeSizeLevel = GEMM_SIZE_NORMAL;
    int mGpuComputeCap = 75;
    int mActivationType = 0;
    std::shared_ptr<Tensor> workspaceTensor;
    void* mWorkspace;
};

} // namespace CUDA
} // namespace MNN

#endif /* ConvInt8CutlassExecution */
#endif