//
//  DeconvSingleInputExecution.hpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DeconvSingleInputExecution_hpp
#define DeconvSingleInputExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "MNNCUDADefine.hpp"
#include "CutlassGemmParam.hpp"
#include "MNNCUDAFunction.cuh"

namespace MNN {
namespace CUDA {

struct IOInfo {
    int ib, ic, ih, iw;
    int ob, oc, oh, ow;
};
struct KernelInfo {
    int groups         = 0;
    int kernelN        = 0;
    int kernelC        = 0;
    int kernelX        = 0;
    int kernelY        = 0;
    int strideX        = 0;
    int strideY        = 0;
    int dilateX        = 0;
    int dilateY        = 0;
    int activationType = 0;
};//

struct Col2ImParameter {
    int padX;
    int padY;
    int dilateX;
    int dilateY;
    int strideX;
    int strideY;
    int kernelX;
    int kernelY;
    int oc;
    int ic;
    int iw;
    int ih;
    int ow;
    int oh;
    int ob;
};

struct InputReorderParameter {
    int ic_stride;
    int ib_stride;
    int oc_stride;
    int ob_stride;
    int hw_size;
    int l_size;
    int h_size;
    int lpack_size;
    int hpack_size;
}; 


extern "C"
class DeconvSingleInputExecution : public Execution {
public:
    struct Resource {
        Resource(Backend* bn, const MNN::Op* op);
        ~ Resource();
        void* mFilter;
        void* mBias;
        std::shared_ptr<Tensor> weightTensor;
        std::shared_ptr<Tensor> biasTensor;
        KernelInfo mKernelInfo;
        Backend* mBackend = nullptr;
    };
    DeconvSingleInputExecution(Backend* backend, const MNN::Op* op,  std::shared_ptr<Resource> res);
    virtual ~DeconvSingleInputExecution();
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<Resource> mResource;
    const Op* mOp = nullptr;
    Col2ImParameter mCol2ImParamter;

    CutlassGemmInfo mGemmInfo;
    int mActivationType;
    int mGpuComputeCap;
    void* mIm2ColBuffer;
    void* mInputBuffer;
    std::shared_ptr<Tensor> workspaceTensor;
    void* mWorkspace;
    void* mZeroPtr;
    std::shared_ptr<Tensor> mZeroTensor;

    bool mFp16Infer = false;
    bool mFp32Infer = false;
    bool mFp16Fp32MixInfer = false;

    GemmCuda_F16_F16_Linear_AlignCuda mGemmCudaF16F16Ln;
    GemmCuda_F16_F32_Linear_AlignCuda mGemmCudaF16F32Ln;
    GemmCuda_F32_F32_Linear_AlignCuda mGemmCudaF32F32Ln;

    GemmCuda_F16_F16_Relu_AlignCuda mGemmCudaF16F16Relu;
    GemmCuda_F16_F32_Relu_AlignCuda mGemmCudaF16F32Relu;
    GemmCuda_F32_F32_Relu_AlignCuda mGemmCudaF32F32Relu;

    GemmCuda_F16_F16_Relu6_AlignCuda mGemmCudaF16F16Relu6;
    GemmCuda_F16_F32_Relu6_AlignCuda mGemmCudaF16F32Relu6;
    GemmCuda_F32_F32_Relu6_AlignCuda mGemmCudaF32F32Relu6;

    GemmTensor_F16_F16_Linear_AlignCuda_Sm75 mGemmF16F16LnSm75;
    GemmTensor_F16_F32_Linear_AlignCuda_Sm75 mGemmF16F32LnSm75;

    GemmTensor_F16_F16_Relu_AlignCuda_Sm75 mGemmF16F16ReluSm75;
    GemmTensor_F16_F32_Relu_AlignCuda_Sm75 mGemmF16F32ReluSm75;

    GemmTensor_F16_F16_Relu6_AlignCuda_Sm75 mGemmF16F16Relu6Sm75;
    GemmTensor_F16_F32_Relu6_AlignCuda_Sm75 mGemmF16F32Relu6Sm75;
};

} // namespace CUDA
} // namespace MNN

#endif /* DeconvSingleInputExecution */
