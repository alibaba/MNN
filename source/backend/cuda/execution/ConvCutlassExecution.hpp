//
//  ConvCutlassExecution.hpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef ConvCutlassExecution_hpp
#define ConvCutlassExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "CutlassGemmParam.hpp"

namespace MNN {
namespace CUDA {

struct CutlassGemmInfo{
    int elh[3];
    int elhPad[3];
};

class ConvCutlassExecution : public Execution {
public:
    struct Resource {
        Resource(Backend* bn, const MNN::Op* op);
        ~ Resource();
        void* mFilter;
        void* mBias;
        std::shared_ptr<Tensor> weightTensor;
        std::shared_ptr<Tensor> biasTensor;
        Backend* mBackend = nullptr;
    };
    ConvCutlassExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res);
    virtual ~ConvCutlassExecution();
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<Resource> mResource;

    const Op* mOp = nullptr;
    CutlassGemmInfo mGemmInfo;

    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    std::pair<void*, int> mGpuIm2ColParam;

    __half* mIm2ColBuffer;

    bool mIsConv1x1S1D1P0 = false;
    bool mNeedIm2Col = true;
    std::pair<void*, int> mGpuKernelParam;
    bool mIsBlock = false;
    int mBlockNum = 1;

    Gemm_F16_Linear_Sm70 mGemmF16LnSm70;
    Gemm_F32_Linear_Sm70 mGemmF32LnSm70;

    Gemm_F16_Relu_Sm70 mGemmF16ReluSm70;
    Gemm_F32_Relu_Sm70 mGemmF32ReluSm70;

    Gemm_F16_Relu6_Sm70 mGemmF16Relu6Sm70;
    Gemm_F32_Relu6_Sm70 mGemmF32Relu6Sm70;

    Gemm_F16_Linear_Sm75 mGemmF16LnSm75;
    Gemm_F32_Linear_Sm75 mGemmF32LnSm75;

    Gemm_F16_Relu_Sm75 mGemmF16ReluSm75;
    Gemm_F32_Relu_Sm75 mGemmF32ReluSm75;

    Gemm_F16_Relu6_Sm75 mGemmF16Relu6Sm75;
    Gemm_F32_Relu6_Sm75 mGemmF32Relu6Sm75;

    int mGpuComputeCap = 75;
    int mActivationType = 0;
    std::shared_ptr<Tensor> workspaceTensor;
    uint8_t* mWorkspace;
};

} // namespace CUDA
} // namespace MNN

#endif /* ConvCutlassExecution */