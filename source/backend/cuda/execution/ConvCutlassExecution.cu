//
//  ConvCutlassExecution.cpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvCutlassExecution.hpp"
#include "Raster.cuh"
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"

//#define DEBUG

namespace MNN {
namespace CUDA {

template<typename T>
__global__ void Im2Col_packC(const ConvolutionCommon::Im2ColParameter* param,
    const size_t maxCount,
    const int iBlock,
    const int pack,
    const int e,
    const int l,
    const T* A,
    half* AP,
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
            *(AP + indexO) = (half)0.0f;
            continue;
        }
        // Compute for source
        int ox, oby, ob, oy, ic, kI, ksx, ksy;
        d_ow.divmod(eIndex, oby, ox);
        d_oh.divmod(oby, ob, oy);
        d_fxy.divmod(lpIndex, ic, kI);
        d_fx.divmod(kI, ksy, ksx);

        size_t sx = ox * param->strideX + ksx * param->dilateX - param->padX;
        size_t sy = oy * param->strideY + ksy * param->dilateY - param->padY;

        const int ic_p = (param->icDiv4) * pack;
        if (sx >= 0 && sx < param->iw) {
            if (sy >=0 && sy < param->ih) {
                size_t offset = ((ob * param->ih + sy) * param->iw + sx) * ic_p + ic;
                *(AP + indexO) = (half)(*(A + offset));
                continue;
            }
        }
        *(AP + indexO) = (half)0.0f;
    }
}

__global__ void WeightPackFill(const float* param,
    half* output,
    const size_t maxCount,
    const int l,
    const int h,
    DivModFast d_lp
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int lpIndex, hpIndex;
        d_lp.divmod(index, hpIndex, lpIndex);

        if(lpIndex >= l || hpIndex >= h) {
            output[index] = (half)0.0f;
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
        float2* srcPtr = (float2 *)(param + (index << 1));
        half2* dstPtr = (half2*)(output + (index << 1));
        dstPtr[0] = __float22half2_rn(srcPtr[0]);
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
        weightTensor.reset(Tensor::createDevice<int16_t>({lp * hp}));
        bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
        mFilter = (void *)weightTensor.get()->buffer().device;

        DivModFast lpD(lp);
        int block_num = runtime->blocks_num(lp*hp);
        int block_size = runtime->threads_num();
        WeightPackFill<<<block_num, block_size>>>((float*)cacheWeight, (half*)mFilter, lp*hp, l, h, lpD);
        checkKernelErrors;

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

            int maxCount = hp / 2;
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
    auto staticPool = static_cast<CUDABackend*>(backend)->getStaticBufferPool();
    mGpuIm2ColParam = staticPool->alloc(sizeof(ConvolutionCommon::Im2ColParameter));
}

ConvCutlassExecution::~ConvCutlassExecution() {
    auto staticPool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
    staticPool->free(mGpuIm2ColParam);

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
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(input);

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

    runtime->memcpy((uint8_t*)mGpuIm2ColParam.first + mGpuIm2ColParam.second, &mIm2ColParamter, sizeof(ConvolutionCommon::Im2ColParameter), MNNMemcpyHostToDevice);

    //MNN_PRINT("conv size:%d-%d, %d-%d-%d, %d-%d-%d\n", mIm2ColParamter.kernelX, mIm2ColParamter.strideX, input->height(), input->width(), input->channel(), output->height(), output->width(), output->channel());
    int e = output->height() * output->width() * output->batch();
    int l = ic * mIm2ColParamter.kernelX * mIm2ColParamter.kernelY;
    int h = output->channel();
    mGemmInfo.elh[0] = e;
    mGemmInfo.elh[1] = l;
    mGemmInfo.elh[2] = h;
    mGemmInfo.elhPad[0] = UP_DIV(e, 8) * 8;
    mGemmInfo.elhPad[1] = UP_DIV(l, 8) * 8;;
    mGemmInfo.elhPad[2] = UP_DIV(h, 8) * 8;;

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
    mNeedIm2Col = !(mIsConv1x1S1D1P0 & (bytes == 2));

    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    if(mNeedIm2Col) {
        auto buffer = pool->alloc((size_t)sizeof(__half) * (size_t)mGemmInfo.elh[0] * (size_t)mGemmInfo.elhPad[1]);
        mIm2ColBuffer = (__half*)((uint8_t*)buffer.first + buffer.second);
        pool->free(buffer);
    }


    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(1);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;
    cutlass::gemm::GemmCoord problem_size(mGemmInfo.elh[0], mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]);// m n k
    //MNN_PRINT("Conv Gemm mnk:%d-%d-%d\n", mGemmInfo.elh[0], mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]);
    ElementInputA *inputA_ptr = mNeedIm2Col ? (ElementInputA *)mIm2ColBuffer : (ElementInputA *)input->deviceId();

    mGpuComputeCap = runtime->compute_capability();
    //MNN_PRINT("Gpu smArch is sm_%d\n", mGpuComputeCap);
    if(mGpuComputeCap < 75) {

        if(mActivationType == 1) {
            if(bytes == 2) {
                // Create a tuple of gemm fp16 + relu kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename Gemm_F16_Relu_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInputB *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = Gemm_F16_Relu_Sm70::get_workspace_size(arguments);
    
                auto buffer3 = pool->alloc(workspace_size * sizeof(uint8_t));
                mWorkspace = (uint8_t*)buffer3.first + buffer3.second;
                runtime->memset(mWorkspace, 0, workspace_size * sizeof(uint8_t));
                pool->free(buffer3);
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16ReluSm70.can_implement(arguments);
                cutlass_check(status);
            
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16ReluSm70.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename Gemm_F32_Relu_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInputB *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = Gemm_F32_Relu_Sm70::get_workspace_size(arguments);
    
                auto buffer3 = pool->alloc(workspace_size * sizeof(uint8_t));
                mWorkspace = (uint8_t*)buffer3.first + buffer3.second;
                runtime->memset(mWorkspace, 0, workspace_size * sizeof(uint8_t));
                pool->free(buffer3);
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF32ReluSm70.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF32ReluSm70.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
    
        } else if(mActivationType == 2) {
    
            if(bytes == 2) {
                // Create a tuple of gemm fp16 + relu6 kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename Gemm_F16_Relu6_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInputB *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = Gemm_F16_Relu6_Sm70::get_workspace_size(arguments);
    
                auto buffer3 = pool->alloc(workspace_size * sizeof(uint8_t));
                mWorkspace = (uint8_t*)buffer3.first + buffer3.second;
                runtime->memset(mWorkspace, 0, workspace_size * sizeof(uint8_t));
                pool->free(buffer3);
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF16Relu6Sm70.can_implement(arguments);
                cutlass_check(status);
            
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16Relu6Sm70.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename Gemm_F32_Relu6_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInputB *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = Gemm_F32_Relu6_Sm70::get_workspace_size(arguments);
    
                auto buffer3 = pool->alloc(workspace_size * sizeof(uint8_t));
                mWorkspace = (uint8_t*)buffer3.first + buffer3.second;
                runtime->memset(mWorkspace, 0, workspace_size * sizeof(uint8_t));
                pool->free(buffer3);
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmF32Relu6Sm70.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF32Relu6Sm70.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
    
        } else {
        
            if(bytes == 2) {
                typename Gemm_F16_Linear_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            {(ElementInputB *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                            {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = Gemm_F16_Linear_Sm70::get_workspace_size(arguments);
    
                auto buffer3 = pool->alloc(workspace_size * sizeof(uint8_t));
                mWorkspace = (uint8_t*)buffer3.first + buffer3.second;
                runtime->memset(mWorkspace, 0, workspace_size * sizeof(uint8_t));
                pool->free(buffer3);
    
                cutlass::Status status = mGemmF16LnSm70.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF16LnSm70.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                typename Gemm_F32_Linear_Sm70::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInputB *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = Gemm_F32_Linear_Sm70::get_workspace_size(arguments);
    
                auto buffer3 = pool->alloc(workspace_size * sizeof(uint8_t));
                mWorkspace = (uint8_t*)buffer3.first + buffer3.second;
                runtime->memset(mWorkspace, 0, workspace_size * sizeof(uint8_t));
                pool->free(buffer3);
    
                cutlass::Status status = mGemmF32LnSm70.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmF32LnSm70.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
        }
        return NO_ERROR;
    }

    if(mActivationType == 1) {
        if(bytes == 2) {
            // Create a tuple of gemm fp16 + relu kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename Gemm_F16_Relu_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInputB *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = Gemm_F16_Relu_Sm75::get_workspace_size(arguments);

            auto buffer3 = pool->alloc(workspace_size * sizeof(uint8_t));
            mWorkspace = (uint8_t*)buffer3.first + buffer3.second;
            runtime->memset(mWorkspace, 0, workspace_size * sizeof(uint8_t));
            pool->free(buffer3);

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF16ReluSm75.can_implement(arguments);
            cutlass_check(status);
        
            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16ReluSm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
            // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename Gemm_F32_Relu_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInputB *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = Gemm_F32_Relu_Sm75::get_workspace_size(arguments);

            auto buffer3 = pool->alloc(workspace_size * sizeof(uint8_t));
            mWorkspace = (uint8_t*)buffer3.first + buffer3.second;
            runtime->memset(mWorkspace, 0, workspace_size * sizeof(uint8_t));
            pool->free(buffer3);

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF32ReluSm75.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF32ReluSm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }

    } else if(mActivationType == 2) {

        if(bytes == 2) {
            // Create a tuple of gemm fp16 + relu6 kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename Gemm_F16_Relu6_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInputB *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = Gemm_F16_Relu6_Sm75::get_workspace_size(arguments);

            auto buffer3 = pool->alloc(workspace_size * sizeof(uint8_t));
            mWorkspace = (uint8_t*)buffer3.first + buffer3.second;
            runtime->memset(mWorkspace, 0, workspace_size * sizeof(uint8_t));
            pool->free(buffer3);

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF16Relu6Sm75.can_implement(arguments);
            cutlass_check(status);
        
            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16Relu6Sm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
            // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename Gemm_F32_Relu6_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInputB *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = Gemm_F32_Relu6_Sm75::get_workspace_size(arguments);

            auto buffer3 = pool->alloc(workspace_size * sizeof(uint8_t));
            mWorkspace = (uint8_t*)buffer3.first + buffer3.second;
            runtime->memset(mWorkspace, 0, workspace_size * sizeof(uint8_t));
            pool->free(buffer3);

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF32Relu6Sm75.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF32Relu6Sm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }

    } else {
    
        if(bytes == 2) {
            typename Gemm_F16_Linear_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                        {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                        {(ElementInputB *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                        {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                        {(ElementOutput_F16 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                        {alpha, beta},          // <- tuple of alpha and beta
                                        split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = Gemm_F16_Linear_Sm75::get_workspace_size(arguments);

            auto buffer3 = pool->alloc(workspace_size * sizeof(uint8_t));
            mWorkspace = (uint8_t*)buffer3.first + buffer3.second;
            runtime->memset(mWorkspace, 0, workspace_size * sizeof(uint8_t));
            pool->free(buffer3);

            cutlass::Status status = mGemmF16LnSm75.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16LnSm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
            typename Gemm_F32_Linear_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {inputA_ptr, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInputB *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)output->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = Gemm_F32_Linear_Sm75::get_workspace_size(arguments);

            auto buffer3 = pool->alloc(workspace_size * sizeof(uint8_t));
            mWorkspace = (uint8_t*)buffer3.first + buffer3.second;
            runtime->memset(mWorkspace, 0, workspace_size * sizeof(uint8_t));
            pool->free(buffer3);

            cutlass::Status status = mGemmF32LnSm75.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF32LnSm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }
    }
    //printf("%d-%d-%d,\n", mGemmInfo.elh[0], mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]);


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
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(input);
    const void *input_addr = (const void*)inputs[0]->deviceId();
    const void *filter_addr = mResource->mFilter;
    const void *bias_addr = mResource->mBias;
    auto bn = backend();
    void *output_addr = (void*)outputs[0]->deviceId();

    auto gpuIm2Col = (const ConvolutionCommon::Im2ColParameter*)((uint8_t*)mGpuIm2ColParam.first + mGpuIm2ColParam.second);

    //printf("%d-%d-%d-%d-%d, %d-%d\n", cpuIm2Col->icDiv4, cpuIm2Col->ih, cpuIm2Col->iw, cpuIm2Col->oh, cpuIm2Col->ow, eAlign, lAlign);
    // Im2col in Block
    for(int block_idx = 0; block_idx < mBlockNum; block_idx++) {
        if(mIsConv1x1S1D1P0 && bytes == 4) {
            size_t maxCount = mGemmInfo.elh[0] * mGemmInfo.elhPad[1] / 2;
            int block_num = runtime->blocks_num(maxCount);
            int block_size = runtime->threads_num();
            Float22Half2<<<block_num, block_size>>>((float*)input_addr, mIm2ColBuffer, maxCount);
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

            if(bytes == 4) {
                Im2Col_packC<<<block_num, block_size>>>(gpuIm2Col, maxCount, block_idx, PACK_NUMBER, mGemmInfo.elh[0], mGemmInfo.elh[1], (const float*)input_addr, mIm2ColBuffer, \
                    lpD, owD, ohD, fxyD, fxD);
                checkKernelErrors;
            } else {
                Im2Col_packC<<<block_num, block_size>>>(gpuIm2Col, maxCount, block_idx, PACK_NUMBER, mGemmInfo.elh[0], mGemmInfo.elh[1], (const half*)input_addr, mIm2ColBuffer, \
                    lpD, owD, ohD, fxyD, fxD);
                checkKernelErrors;
            }
        }
    }

    if(mGpuComputeCap < 75) {
        if(mActivationType == 1) {
            if(bytes == 4) {
                cutlass::Status status = mGemmF32ReluSm70();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmF16ReluSm70();
                cutlass_check(status);
            }
        } else if(mActivationType == 2) {
            if(bytes == 4) {
                cutlass::Status status = mGemmF32Relu6Sm70();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmF16Relu6Sm70();
                cutlass_check(status);
            }
        } else {
            if(bytes == 4) {
                cutlass::Status status = mGemmF32LnSm70();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmF16LnSm70();
                cutlass_check(status);
            }
        }
    
        return NO_ERROR;
    }

    if(mActivationType == 1) {
        if(bytes == 4) {
            cutlass::Status status = mGemmF32ReluSm75();
            cutlass_check(status);
        } else {
            cutlass::Status status = mGemmF16ReluSm75();
            cutlass_check(status);
        }
    } else if(mActivationType == 2) {
        if(bytes == 4) {
            cutlass::Status status = mGemmF32Relu6Sm75();
            cutlass_check(status);
        } else {
            cutlass::Status status = mGemmF16Relu6Sm75();
            cutlass_check(status);
        }
    } else {
        if(bytes == 4) {
            cutlass::Status status = mGemmF32LnSm75();
            cutlass_check(status);
        } else {
            cutlass::Status status = mGemmF16LnSm75();
            cutlass_check(status);
        }
    }

    return NO_ERROR;
}


}// namespace CUDA
}// namespace MNN