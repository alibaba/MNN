//
//  ConvImplicitExecution.hpp
//  MNN
//
//  Created by MNN on 2024/01/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef ConvImplicitExecution_hpp_
#define ConvImplicitExecution_hpp_

#include "ConvSingleInputExecution.hpp"
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"
#include "CutlassGemmParam.hpp"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#ifdef ENABLE_CUDA_TUNE_PARAM
#include "cutlass_common/tune/CutlassGemmTuneCommonExecution.hpp"
#endif
namespace MNN {
namespace CUDA {

using Layout_NHWC = cutlass::layout::TensorNHWC;
using Layout_NHWC = cutlass::layout::TensorNHWC;
using Layout_NHWC = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 32>;
using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

//using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>128
constexpr int NumStagesSm80 = 2;

cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized; // Which iterator algorithm to use: Analytic or Optimized

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput_F16,                                     // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput_F16>::value,  // The number of elements per vectorized
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInput_F16, Layout_NHWC,
  ElementInput_F16, Layout_NHWC,
  ElementOutput_F16, Layout_NHWC,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStagesSm80,
  cutlass::arch::OpMultiplyAdd,
  IteratorAlgorithm
>::Kernel;

using ImplicitConv = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;


class ConvImplicitExecution : 
    #ifdef ENABLE_CUDA_TUNE_PARAM
    public CutlassGemmTuneCommonExecution
    #else
    public Execution 
    #endif
{
public:
    struct Resource;
    static bool isValid(const Convolution2D* conv, const Tensor* input, const Tensor* output, Backend* backend);
    ConvImplicitExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res);
    virtual ~ConvImplicitExecution();

    struct Resource {
        Resource(Backend* backend, const MNN::Op* op);
        ~ Resource();
        void* mFilter;
        void* mBias;
        std::shared_ptr<Tensor> weightTensor;
        std::shared_ptr<Tensor> biasTensor;
        KernelInfo mKernelInfo;
        Backend* mBackend = nullptr;
    };

    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    
private:
    std::shared_ptr<Resource> mResource;
    const Op* mOp = nullptr;
    void* mBtdB_Buffer;
    void* mMatmul_Buffer;

    ImplicitConv mImplicitConvOp;

    std::shared_ptr<Tensor> workspaceTensor;
    void* mWorkspace;

    int mPadX;
    int mPadY;
    int mBlock2;
    int mGpuComputeCap;
    bool mIsTuned =false;
    int mActivationType;
    bool mFp16Infer = false;
    bool mFp32Infer = false;
    bool mFp16Fp32MixInfer = false;
};

} // namespace CUDA
} // namespace MNN
#endif
