//
//  ConvCutlassExecution.cpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvCutlassExecution.hpp"
#include "Raster.cuh"

//#define DEBUG

namespace MNN {
namespace CUDA {

template<typename T0, typename T1>
__global__ void Im2Col_packC(
    const int sw,
    const int sh,
    const int dw,
    const int dh,
    const int pw,
    const int ph,
    const int icDiv4,
    const int iw,
    const int ih,
    const size_t maxCount,
    const int iBlock,
    const int pack,
    const int e,
    const int l,
    const T0* A,
    T1* AP,
    DivModFast d_lp,
    DivModFast d_ow,
    DivModFast d_oh,
    DivModFast d_fxy,
    DivModFast d_fx
) {

    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int eIndex, lpIndex;
        d_lp.divmod(indexO, eIndex, lpIndex);

        if(eIndex >= e || lpIndex >= l) {
            *(AP + indexO) = (T1)0.0f;
            continue;
        }
        // Compute for source
        int ox, oby, ob, oy, ic, kI, ksx, ksy;
        d_ow.divmod(eIndex, oby, ox);
        d_oh.divmod(oby, ob, oy);
        d_fxy.divmod(lpIndex, ic, kI);
        d_fx.divmod(kI, ksy, ksx);

        size_t sx = ox * sw + ksx * dw - pw;
        size_t sy = oy * sh + ksy * dh- ph;

        const int ic_p = icDiv4 * pack;
        if (sx >= 0 && sx < iw) {
            if (sy >=0 && sy < ih) {
                size_t offset = ((ob * ih + sy) * iw + sx) * ic_p + ic;
                *(AP + indexO) = (T1)(*(A + offset));
                continue;
            }
        }
        *(AP + indexO) = (T1)0.0f;
    }
}

template<typename T>
__global__ void WeightPackFill(const float* param,
    T* output,
    const size_t maxCount,
    const int l,
    const int h,
    DivModFast d_lp
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int lpIndex, hpIndex;
        d_lp.divmod(index, hpIndex, lpIndex);

        if(lpIndex >= l || hpIndex >= h) {
            output[index] = (T)0.0f;
            continue;
        }
        output[index] = param[hpIndex * l + lpIndex];
    }
}

__global__ void Float22Half2(const float* param,
    half* output,
    const size_t maxCount
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        float2* srcPtr = (float2 *)(param + (index << 2));
        half2* dstPtr = (half2*)(output + (index << 2));
        dstPtr[0] = __float22half2_rn(srcPtr[0]);
        dstPtr[1] = __float22half2_rn(srcPtr[1]);
    }
}

ConvCutlassExecution::Resource::Resource(Backend* bn, const MNN::Op* op) {
    mBackend = bn;
    auto runtime = static_cast<CUDABackend*>(bn)->getCUDARuntime();

    auto conv       = op->main_as_Convolution2D();
    auto common     = conv->common();

    //weight host->device
    const float* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, conv, &filterDataPtr, &weightSize);
    auto oc = common->outputCount();

    int l = weightSize / oc;
    int h = oc;
    int lp = UP_DIV(l, 8) * 8;
    int hp = UP_DIV(h, 8) * 8;

    // Reorder weight
    {
        auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(float));
        float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
        runtime->memcpy(cacheWeight, filterDataPtr, weightSize * sizeof(float), MNNMemcpyHostToDevice);
        if(static_cast<CUDABackend*>(bn)->getPrecision() == 1) {
            weightTensor.reset(Tensor::createDevice<int32_t>({lp * hp}));
        } else {
            weightTensor.reset(Tensor::createDevice<int16_t>({lp * hp}));
        }
        bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
        mFilter = (void *)weightTensor.get()->buffer().device;

        DivModFast lpD(lp);
        int block_num = runtime->blocks_num(lp*hp);
        int block_size = runtime->threads_num();
         // Only when fp32 Weight convert to fp32, Fp16Fp32Mix Weight convert to fp16
        if(static_cast<CUDABackend*>(bn)->getPrecision() == 1) {
            WeightPackFill<<<block_num, block_size>>>((float*)cacheWeight, (float*)mFilter, lp*hp, l, h, lpD);
            checkKernelErrors;
        } else {
            WeightPackFill<<<block_num, block_size>>>((float*)cacheWeight, (half*)mFilter, lp*hp, l, h, lpD);
            checkKernelErrors;
        }

        static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempCacheBuffer);
    }

    // Copy Bias
    {
        if(static_cast<CUDABackend*>(bn)->useFp16()) {
            auto tempBiasStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(conv->bias()->size()*sizeof(float));
            auto biasTemp = (float*)((uint8_t*)tempBiasStorage.first + tempBiasStorage.second);
            cuda_check(cudaMemcpy(biasTemp, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));

            int biasSize = conv->bias()->size();
            int hp = UP_DIV(biasSize, 8) * 8;
            biasTensor.reset(Tensor::createDevice<int16_t>({hp}));
            bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
            mBias = (void *)biasTensor.get()->buffer().device;
            runtime->memset(mBias, 0, hp * sizeof(int16_t));

            int maxCount = hp / 4;
            int block_num = runtime->blocks_num(maxCount);
            int block_size = runtime->threads_num();
            Float22Half2<<<block_num, block_size>>>((float*)biasTemp, (half*)mBias, maxCount);
            checkKernelErrors;

            static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempBiasStorage);
        } else {
            int biasSize = conv->bias()->size();
            int hp = UP_DIV(biasSize, 8) * 8;
            biasTensor.reset(Tensor::createDevice<int32_t>({hp}));
            bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
            mBias = (void *)biasTensor.get()->buffer().device;
            runtime->memset(mBias, 0, hp * sizeof(int32_t));
            cuda_check(cudaMemcpy(mBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));
        }
    }
}

ConvCutlassExecution::Resource::~Resource() {
    // Do nothing
}
ConvCutlassExecution::ConvCutlassExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res) : Execution(backend), mOp(op) {
    mResource = res;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    int precisonLevel = static_cast<CUDABackend*>(backend)->getPrecision();
    mFp16Infer = (precisonLevel == 2);
    mFp32Infer = (precisonLevel == 1);
    mFp16Fp32MixInfer = (precisonLevel == 0);
}

ConvCutlassExecution::~ConvCutlassExecution() {

}
bool ConvCutlassExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvCutlassExecution(bn, op, mResource);
    *dst = dstExe;
    return true;
}


ErrorCode ConvCutlassExecution::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input = inputs[0], output = outputs[0];
    const int UNIT = PACK_NUMBER;
    auto convCommon = mOp->main_as_Convolution2D()->common();
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mOp->main_as_Convolution2D()->common());
    int ic = input->channel();
    auto icDiv = UP_DIV(ic, UNIT);

    mIm2ColParamter.dilateX         = convCommon->dilateX();
    mIm2ColParamter.dilateY         = convCommon->dilateY();
    mIm2ColParamter.strideX         = convCommon->strideX();
    mIm2ColParamter.strideY         = convCommon->strideY();
    mIm2ColParamter.icDiv4          = icDiv;
    mIm2ColParamter.kernelX         = convCommon->kernelX();
    mIm2ColParamter.kernelY         = convCommon->kernelY();
    mIm2ColParamter.padX = std::get<0>(pads);
    mIm2ColParamter.padY = std::get<1>(pads);

    mIm2ColParamter.ih = input->height();
    mIm2ColParamter.iw = input->width();
    mIm2ColParamter.oh = output->height();
    mIm2ColParamter.ow = output->width();
    mIm2ColParamter.srcZStep = input->height() * input->width() * UNIT * input->batch();
    mIm2ColParamter.srcYStep = input->width() * UNIT;
    mIm2ColParamter.packCUnit = UNIT;

    mActivationType = convCommon->relu() ? 1 : convCommon->relu6() ? 2 : 0;

    //MNN_PRINT("conv size:%d-%d, %d-%d-%d, %d-%d-%d\n", mIm2ColParamter.kernelX, mIm2ColParamter.strideX, input->height(), input->width(), input->channel(), output->height(), output->width(), output->channel());
    int e = output->height() * output->width() * output->batch();
    int l = ic * mIm2ColParamter.kernelX * mIm2ColParamter.kernelY;
    int h = output->channel();
    mGemmInfo.elh[0] = e;
    mGemmInfo.elh[1] = l;
    mGemmInfo.elh[2] = h;
    mGemmInfo.elhPad[0] = UP_DIV(e, 8) * 8;
    mGemmInfo.elhPad[1] = UP_DIV(l, 8) * 8;
    mGemmInfo.elhPad[2] = UP_DIV(h, 8) * 8;

    //MNN_PRINT("Activate:%d \n", mActivationType);
    //MNN_PRINT("Im2Col：%d-%d-%d temp size:%zu!!!\n\n",output->width(), ic, mIm2ColParamter.kernelX, (size_t)sizeof(__half) * mMatMulParam.elhPack[0] * mMatMulParam.elhPack[1] * MATMULPACK * MATMULPACK);
    // When Im2Col memory size big than 2GB
    if(0){//(size_t)mGemmInfo.elh[0] * (size_t)mGemmInfo.elh[1] > 1024*1024*1024 && mIm2ColParamter.kernelX > 1 && mIm2ColParamter.kernelY > 1) {
        //printf("need im2col in block\n");
        mIsBlock = true;
        mBlockNum = 16;
        mGemmInfo.elh[0] = UP_DIV(mGemmInfo.elh[0], mBlockNum);
    }

    mIsConv1x1S1D1P0 = (mIm2ColParamter.kernelX == 1 && mIm2ColParamter.kernelY == 1 && \
                        mIm2ColParamter.strideX == 1 && mIm2ColParamter.strideY == 1 && \
                        mIm2ColParamter.dilateX == 1 && mIm2ColParamter.dilateY == 1 && \
                        mIm2ColParamter.padX == 0 && mIm2ColParamter.padY == 0);
    mNeedIm2Col = !(mIsConv1x1S1D1P0 && (mFp16Infer || mFp32Infer));

    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    if(mNeedIm2Col) {
        size_t im2colBytes = 2;
        // Only when fp32 Im2Col convert to fp32, Fp16Fp32Mix Im2Col convert to fp16
        if(mFp32Infer) {
            im2colBytes = 4;
        }
        auto buffer = pool->alloc(im2colBytes * (size_t)mGemmInfo.elh[0] * (size_t)mGemmInfo.elhPad[1]);
        mIm2ColBuffer = (void*)((uint8_t*)buffer.first + buffer.second);
        pool->free(buffer);
    }


    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(1);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;
    cutlass::gemm::GemmCoord problem_size(mGemmInfo.elh[0], mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]);// m n k

    // Float32 inference param
    if(mFp32Infer) {
        ElementInput_F32 *input_fp32_addr = mNeedIm2Col ? (ElementInput_F32 *)mIm2ColBuffer : (ElementInput_F32 *)input->deviceId();
        if(mActivationType == 1) {
            // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmCuda_F32_F32_Relu_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {input_fp32_addr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F32 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmCuda_F32_F32_Relu_AlignCuda::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmCudaF32F32Relu.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmCudaF32F32Relu.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);

        } else if(mActivationType == 2) {
            // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmCuda_F32_F32_Relu6_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {input_fp32_addr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F32 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmCuda_F32_F32_Relu6_AlignCuda::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmCudaF32F32Relu6.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmCudaF32F32Relu6.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
    
        } else {
            typename GemmCuda_F32_F32_Linear_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {input_fp32_addr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F32 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmCuda_F32_F32_Linear_AlignCuda::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }
            cutlass::Status status = mGemmCudaF32F32Ln.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmCudaF32F32Ln.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }
        return NO_ERROR;
    }

    //MNN_PRINT("Conv Gemm mnk:%d-%d-%d\n", mGemmInfo.elh[0], mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]);
    ElementInput_F16 *inputA_ptr = mNeedIm2Col ? (ElementInput_F16 *)mIm2ColBuffer : (ElementInput_F16 *)input->deviceId();

    mGpuComputeCap = runtime->compute_capability();
    //MNN_PRINT("Gpu smArch is sm_%d\n", mGpuComputeCap);
    if(mGpuComputeCap < 70) {

        if(mActivationType == 1) {
            if(mFp16Infer) {
                // Create a tuple of gemm fp16 + relu kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename GemmCuda_F16_F16_Relu_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmCuda_F16_F16_Relu_AlignCuda::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmCudaF16F16Relu.can_implement(arguments);
                cutlass_check(status);
            
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmCudaF16F16Relu.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename GemmCuda_F16_F32_Relu_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmCuda_F16_F32_Relu_AlignCuda::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmCudaF16F32Relu.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmCudaF16F32Relu.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
    
        } else if(mActivationType == 2) {
    
            if(mFp16Infer) {
                // Create a tuple of gemm fp16 + relu6 kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename GemmCuda_F16_F16_Relu6_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmCuda_F16_F16_Relu6_AlignCuda::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmCudaF16F16Relu6.can_implement(arguments);
                cutlass_check(status);
            
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmCudaF16F16Relu6.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename GemmCuda_F16_F32_Relu6_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmCuda_F16_F32_Relu6_AlignCuda::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmCudaF16F32Relu6.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmCudaF16F32Relu6.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
        } else {
        
            if(mFp16Infer) {
                typename GemmCuda_F16_F16_Linear_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                            {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmCuda_F16_F16_Linear_AlignCuda::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                cutlass::Status status = mGemmCudaF16F16Ln.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmCudaF16F16Ln.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                typename GemmCuda_F16_F32_Linear_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmCuda_F16_F32_Linear_AlignCuda::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                cutlass::Status status = mGemmCudaF16F32Ln.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmCudaF16F32Ln.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
        }
        return NO_ERROR;
    } else if(mGpuComputeCap < 75) {

        if(mActivationType == 1) {
            if(mFp16Infer) {
                // Create a tuple of gemm fp16 + relu kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename GemmTensor_F16_F16_Relu_AlignTensor_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmTensor_F16_F16_Relu_AlignTensor_Sm70::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16ReluSm70.can_implement(arguments);
                cutlass_check(status);
            
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16ReluSm70.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename GemmTensor_F16_F32_Relu_AlignTensor_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmTensor_F16_F32_Relu_AlignTensor_Sm70::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32ReluSm70.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32ReluSm70.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
    
        } else if(mActivationType == 2) {
    
            if(mFp16Infer) {
                // Create a tuple of gemm fp16 + relu6 kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename GemmTensor_F16_F16_Relu6_AlignTensor_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmTensor_F16_F16_Relu6_AlignTensor_Sm70::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F16Relu6Sm70.can_implement(arguments);
                cutlass_check(status);
            
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16Relu6Sm70.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename GemmTensor_F16_F32_Relu6_AlignTensor_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmTensor_F16_F32_Relu6_AlignTensor_Sm70::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16F32Relu6Sm70.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32Relu6Sm70.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
    
        } else {
        
            if(mFp16Infer) {
                typename GemmTensor_F16_F16_Linear_AlignTensor_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                            {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmTensor_F16_F16_Linear_AlignTensor_Sm70::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                cutlass::Status status = mGemmF16F16LnSm70.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F16LnSm70.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                typename GemmTensor_F16_F32_Linear_AlignTensor_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmTensor_F16_F32_Linear_AlignTensor_Sm70::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                cutlass::Status status = mGemmF16F32LnSm70.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16F32LnSm70.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
        }
        return NO_ERROR;
    }

    if(mActivationType == 1) {
        if(mFp16Infer) {
            // Create a tuple of gemm fp16 + relu kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmTensor_F16_F16_Relu_AlignTensor_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F16_Relu_AlignTensor_Sm75::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF16F16ReluSm75.can_implement(arguments);
            cutlass_check(status);
        
            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F16ReluSm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
            // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmTensor_F16_F32_Relu_AlignTensor_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F32_Relu_AlignTensor_Sm75::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF16F32ReluSm75.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F32ReluSm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }

    } else if(mActivationType == 2) {

        if(mFp16Infer) {
            // Create a tuple of gemm fp16 + relu6 kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmTensor_F16_F16_Relu6_AlignTensor_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F16_Relu6_AlignTensor_Sm75::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF16F16Relu6Sm75.can_implement(arguments);
            cutlass_check(status);
        
            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F16Relu6Sm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
            // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmTensor_F16_F32_Relu6_AlignTensor_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F32_Relu6_AlignTensor_Sm75::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF16F32Relu6Sm75.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F32Relu6Sm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }

    } else {
    
        if(mFp16Infer) {
            typename GemmTensor_F16_F16_Linear_AlignTensor_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                        {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                        {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                        {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                        {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                        {alpha, beta},          // <- tuple of alpha and beta
                                        split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F16_Linear_AlignTensor_Sm75::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            cutlass::Status status = mGemmF16F16LnSm75.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F16LnSm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
            typename GemmTensor_F16_F32_Linear_AlignTensor_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F32_Linear_AlignTensor_Sm75::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            cutlass::Status status = mGemmF16F32LnSm75.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F32LnSm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }
    }

    return NO_ERROR;
}

ErrorCode ConvCutlassExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    //MNN_PRINT("cuda convSingleInput onExecute in, inputsize:%d %d\n", (int)inputs.size(), workspace_size_);
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];

    //printf("convcutlass:%p %p\n", input->deviceId(), output->deviceId());
    //MNN_PRINT("cutlass hw:%d-%d\n", input->height(), input->width());
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    const void *input_addr = (const void*)inputs[0]->deviceId();
    const void *filter_addr = mResource->mFilter;
    const void *bias_addr = mResource->mBias;
    auto bn = backend();
    void *output_addr = (void*)outputs[0]->deviceId();

    const int sw = mIm2ColParamter.strideX;
    const int sh = mIm2ColParamter.strideY;
    const int dw = mIm2ColParamter.dilateX;
    const int dh = mIm2ColParamter.dilateY;
    const int pw = mIm2ColParamter.padX;
    const int ph = mIm2ColParamter.padY;
    const int icDiv4 = mIm2ColParamter.icDiv4;
    const int iw = mIm2ColParamter.iw;
    const int ih = mIm2ColParamter.ih;

    //printf("%d-%d-%d-%d-%d, %d-%d\n", cpuIm2Col->icDiv4, cpuIm2Col->ih, cpuIm2Col->iw, cpuIm2Col->oh, cpuIm2Col->ow, eAlign, lAlign);
    // Im2col in Block
    for(int block_idx = 0; block_idx < mBlockNum; block_idx++) {
        if(mIsConv1x1S1D1P0 && mFp16Fp32MixInfer) {
            size_t maxCount = mGemmInfo.elh[0] * mGemmInfo.elhPad[1] / 4;
            int block_num = runtime->blocks_num(maxCount);
            int block_size = runtime->threads_num();
            Float22Half2<<<block_num, block_size>>>((float*)input_addr, (half *)mIm2ColBuffer, maxCount);
            checkKernelErrors;
        } else if (mNeedIm2Col) {
            DivModFast lpD(mGemmInfo.elhPad[1]);
            DivModFast fxyD((mIm2ColParamter.kernelX * mIm2ColParamter.kernelY));
            DivModFast fxD(mIm2ColParamter.kernelX);
            DivModFast owD(mIm2ColParamter.ow);
            DivModFast ohD(mIm2ColParamter.oh);
        
            size_t maxCount = mGemmInfo.elh[0] * mGemmInfo.elhPad[1];
            size_t block_num = runtime->blocks_num(maxCount);
            size_t block_size = runtime->threads_num();

            if(mFp32Infer) {
                Im2Col_packC<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih,
                    maxCount, block_idx, PACK_NUMBER, mGemmInfo.elh[0], mGemmInfo.elh[1], (const float*)input_addr, (float *)mIm2ColBuffer, \
                    lpD, owD, ohD, fxyD, fxD);
                checkKernelErrors;
            } else if(mFp16Fp32MixInfer) {
                Im2Col_packC<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, 
                    maxCount, block_idx, PACK_NUMBER, mGemmInfo.elh[0], mGemmInfo.elh[1], (const float*)input_addr, (half *)mIm2ColBuffer, \
                    lpD, owD, ohD, fxyD, fxD);
                checkKernelErrors;
            } else {
                Im2Col_packC<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, 
                    maxCount, block_idx, PACK_NUMBER, mGemmInfo.elh[0], mGemmInfo.elh[1], (const half*)input_addr, (half *)mIm2ColBuffer, \
                    lpD, owD, ohD, fxyD, fxD);
                checkKernelErrors;
            }
        }
    }

    if(mFp32Infer) {
        if(mActivationType == 1) {
            cutlass::Status status = mGemmCudaF32F32Relu();
            cutlass_check(status);
        } else if(mActivationType == 2) {
            cutlass::Status status = mGemmCudaF32F32Relu6();
            cutlass_check(status);
        } else {
            cutlass::Status status = mGemmCudaF32F32Ln();
            cutlass_check(status);
        }
        return NO_ERROR;
    }

    if(mGpuComputeCap < 70) {
        if(mActivationType == 1) {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmCudaF16F32Relu();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmCudaF16F16Relu();
                cutlass_check(status);
            }
        } else if(mActivationType == 2) {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmCudaF16F32Relu6();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmCudaF16F16Relu6();
                cutlass_check(status);
            }
        } else {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmCudaF16F32Ln();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmCudaF16F16Ln();
                cutlass_check(status);
            }
        }
    
        return NO_ERROR;
    } else if(mGpuComputeCap < 75) {
        if(mActivationType == 1) {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmF16F32ReluSm70();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmF16F16ReluSm70();
                cutlass_check(status);
            }
        } else if(mActivationType == 2) {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmF16F32Relu6Sm70();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmF16F16Relu6Sm70();
                cutlass_check(status);
            }
        } else {
            if(mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmF16F32LnSm70();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmF16F16LnSm70();
                cutlass_check(status);
            }
        }
    
        return NO_ERROR;
    }

    if(mActivationType == 1) {
        if(mFp16Fp32MixInfer) {
            cutlass::Status status = mGemmF16F32ReluSm75();
            cutlass_check(status);
        } else {
            cutlass::Status status = mGemmF16F16ReluSm75();
            cutlass_check(status);
        }
    } else if(mActivationType == 2) {
        if(mFp16Fp32MixInfer) {
            cutlass::Status status = mGemmF16F32Relu6Sm75();
            cutlass_check(status);
        } else {
            cutlass::Status status = mGemmF16F16Relu6Sm75();
            cutlass_check(status);
        }
    } else {
        if(mFp16Fp32MixInfer) {
            cutlass::Status status = mGemmF16F32LnSm75();
            cutlass_check(status);
        } else {
            cutlass::Status status = mGemmF16F16LnSm75();
            cutlass_check(status);
        }
    }

    return NO_ERROR;
}


}// namespace CUDA
}// namespace MNN