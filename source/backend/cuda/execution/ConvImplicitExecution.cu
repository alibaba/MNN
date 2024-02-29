//
//  ConvImplicitExecution.cpp
//  MNN
//
//  Created by MNN on 2024/01/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "ConvImplicitExecution.hpp"
#include "ConvBaseKernel.cuh"

namespace MNN {
namespace CUDA {

bool ConvImplicitExecution::isValid(const Convolution2D* conv, const Tensor* input, const Tensor* output, Backend* backend) {
    if (static_cast<CUDABackend*>(backend)->getPrecision() != 2) {
    	return false;
    }
    if(static_cast<CUDABackend*>(backend)->getCUDARuntime()->compute_capability() < 80) {
        return false;
    }
    if(conv->common()->strideX() != 1 || conv->common()->strideY() != 1) {
        return false;
    }
    if(conv->common()->dilateX() != 1 || conv->common()->dilateY() != 1) {
        return false;
    }
    if(conv->common()->kernelX() > 32 || conv->common()->kernelY() > 32) {
        return false;
    }
    if(conv->common()->kernelX() == 1 && conv->common()->kernelY() == 1 && conv->common()->padX() == 0 && conv->common()->padY() == 0) {
        return false;
    }
    if(conv->common()->relu() || conv->common()->relu6()) {
        return false;
    }
    if(input->channel() >= 64 && output->channel() >= 64) {
        return false;
    }
    return true;
}

template<typename T>
__global__ void WeightPackFill_Implicit(const float* param,
    T* output,
    const int khw,
    const size_t maxCount,
    const int ci,
    const int co,
    DivModFast d_cip,
    DivModFast d_khw
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int copIndex, hpIndex, cipIndex, khwIndex;
        d_cip.divmod(index, hpIndex, cipIndex);
        d_khw.divmod(hpIndex, copIndex, khwIndex);

        if(cipIndex >= ci || copIndex >= co) {
            output[index] = (T)0.0f;
            continue;
        }

        // [Co, Ci, KhKw] -> [Cop, KhKw, Cip]
        output[index] = param[(copIndex * ci + cipIndex) * khw + khwIndex];
    }
}

ConvImplicitExecution::Resource::Resource(Backend* backend, const MNN::Op* op) {
    mBackend = backend;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();

    auto conv       = op->main_as_Convolution2D();
    auto common     = conv->common();
    mKernelInfo.kernelX        = common->kernelX();
    mKernelInfo.kernelY        = common->kernelY();
    mKernelInfo.groups         = common->group();
    mKernelInfo.strideX        = common->strideX();
    mKernelInfo.strideY        = common->strideY();
    mKernelInfo.dilateX        = common->dilateX();
    mKernelInfo.dilateY        = common->dilateY();

    //weight host->device
    const float* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, backend, conv, &filterDataPtr, &weightSize);
    mKernelInfo.kernelN = common->outputCount();
    mKernelInfo.kernelC = weightSize / mKernelInfo.kernelN / mKernelInfo.kernelX / mKernelInfo.kernelY;

    auto dstWeightSize = weightSize;

    // Reorder weight
    {
        int ci_pack = UP_DIV(mKernelInfo.kernelC, PACK_NUMBER) * PACK_NUMBER;
        int co_pack = UP_DIV(mKernelInfo.kernelN, PACK_NUMBER) * PACK_NUMBER;
        int khw = mKernelInfo.kernelX * mKernelInfo.kernelY;
    
        auto tempCacheBuffer = static_cast<CUDABackend*>(backend)->getStaticBufferPool()->alloc(weightSize * sizeof(float));
        float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
        runtime->memcpy(cacheWeight, filterDataPtr, weightSize * sizeof(float), MNNMemcpyHostToDevice);
        if(static_cast<CUDABackend*>(backend)->getPrecision() == 1) {
            weightTensor.reset(Tensor::createDevice<int32_t>({ci_pack * co_pack * khw}));
        } else {
            weightTensor.reset(Tensor::createDevice<int16_t>({ci_pack * co_pack * khw}));
        }
        backend->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
        mFilter = (void *)weightTensor.get()->buffer().device;


        DivModFast cipD(ci_pack);
        DivModFast khwD(khw);
    
        int block_num = runtime->blocks_num(ci_pack * co_pack * khw);
        int block_size = runtime->threads_num();
    
        if(static_cast<CUDABackend*>(backend)->getPrecision() == 1) {
            WeightPackFill_Implicit<<<block_num, block_size>>>((const float*)cacheWeight, (float*)mFilter, khw, ci_pack * co_pack * khw, mKernelInfo.kernelC, mKernelInfo.kernelN, cipD, khwD);
            checkKernelErrors;
        } else {
            WeightPackFill_Implicit<<<block_num, block_size>>>((const float*)cacheWeight, (half*)mFilter, khw, ci_pack * co_pack * khw, mKernelInfo.kernelC, mKernelInfo.kernelN, cipD, khwD);
            checkKernelErrors;            
        }
        static_cast<CUDABackend*>(backend)->getStaticBufferPool()->free(tempCacheBuffer);
    }

    if(static_cast<CUDABackend*>(backend)->useFp16()) {
        int biasSize = conv->bias()->size();
        int hp = UP_DIV(biasSize, PACK_NUMBER) * PACK_NUMBER;

        auto tempBiasStorage = static_cast<CUDABackend*>(backend)->getStaticBufferPool()->alloc(hp*sizeof(float));
        auto biasTemp = (float*)((uint8_t*)tempBiasStorage.first + tempBiasStorage.second);
        runtime->memset(biasTemp, 0, hp * sizeof(int32_t));
        cuda_check(cudaMemcpy(biasTemp, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));

        biasTensor.reset(Tensor::createDevice<int16_t>({hp}));
        backend->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
        mBias = (void *)biasTensor.get()->buffer().device;
        callFloat2Half((const void*)biasTemp, (void*)mBias, hp, runtime);

        static_cast<CUDABackend*>(backend)->getStaticBufferPool()->free(tempBiasStorage);
    } else {
        int biasSize = conv->bias()->size();
        int alignSize = UP_DIV(biasSize, PACK_NUMBER) * PACK_NUMBER;
        biasTensor.reset(Tensor::createDevice<uint32_t>({alignSize}));
        backend->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
    
        mBias = (void *)biasTensor.get()->buffer().device;
        cuda_check(cudaMemset(mBias, 0, alignSize*sizeof(float)));
        cuda_check(cudaMemcpy(mBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));
      }
}

ConvImplicitExecution::Resource::~Resource() {
    // Do nothing
}

ConvImplicitExecution::ConvImplicitExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res) :
    #ifdef ENABLE_CUDA_TUNE_PARAM
    CutlassGemmTuneCommonExecution(backend),
    #else
    Execution(backend),
    #endif
    mOp(op) 
{
    mResource = res;
    int precisonLevel = static_cast<CUDABackend*>(backend)->getPrecision();
    mFp16Infer = (precisonLevel == 2);
    mFp32Infer = (precisonLevel == 1);
    mFp16Fp32MixInfer = (precisonLevel == 0);
}
ConvImplicitExecution::~ConvImplicitExecution() {
    // Nothing
}

bool ConvImplicitExecution::onClone(Backend* backend, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if(nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvImplicitExecution(backend, op, mResource);
    *dst = dstExe;
    return true;
}

ErrorCode ConvImplicitExecution::onResize(const std::vector<Tensor*>  &inputs, const std::vector<Tensor*> &outputs) {

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    auto input = inputs[0];
    auto output = outputs[0];
    auto convCommon = mOp->main_as_Convolution2D()->common();
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mOp->main_as_Convolution2D()->common());
    mPadX = std::get<0>(pads);
    mPadY = std::get<1>(pads);
    int ic = input->channel();
    int icDiv = UP_DIV(ic, PACK_NUMBER);
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(input);

    mActivationType = convCommon->relu() ? 1 : convCommon->relu6() ? 2 : 0;
    //MNN_PRINT("!!conv size:3-1, %d-%d-%d, %d-%d-%d\n", input->height(), input->width(), input->channel(), output->height(), output->width(), output->channel());


    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    // Split K dimension into 1 partitions
    int split_k_slices = 1;
    int ci_pack = UP_DIV(input->channel(), PACK_NUMBER) * PACK_NUMBER;
    int co_pack = UP_DIV(output->channel(), PACK_NUMBER) * PACK_NUMBER;  
    // Construct Conv2dProblemSize with user defined output size
    cutlass::conv::Conv2dProblemSize problem_size(
        input->batch(),//int N,
        input->height(),//int H,
        input->width(),//int W,
        ci_pack,//int C,
        co_pack,//int K,
        mResource->mKernelInfo.kernelY,//int R,
        mResource->mKernelInfo.kernelX,//int S,
        output->height(),//int P,
        output->width(),//int Q,
        mPadY,//int pad_h,
        mPadX,//int pad_w,
        mResource->mKernelInfo.strideY,//int stride_h,
        mResource->mKernelInfo.strideX,//int stride_w,
        mResource->mKernelInfo.dilateY,//int dilation_h,
        mResource->mKernelInfo.dilateX,//int dilation_w,
        mode,//Mode mode,
        split_k_slices,//int split_k_slices = 1,
        1//int groups = 1
    );
    // printf("%d %d %d %d, %d %d %d, %d %d, %d %d %d %d %d %d\n", input->batch(), input->height(), input->width(), input->channel(), mResource->mKernelInfo.kernelN, mResource->mKernelInfo.kernelY, mResource->mKernelInfo.kernelX, output->height(), output->width(), mPadY, mPadX, mResource->mKernelInfo.strideY, mResource->mKernelInfo.strideX, mResource->mKernelInfo.dilateY, mResource->mKernelInfo.dilateX);

    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(1);

    // Construct ImplicitConv::Argument structure with conv2d
    // problem size, data pointers, and epilogue values
    typename ImplicitConv::Arguments arguments{
      problem_size,
      {(ElementInput_F16 *)input->deviceId(), {ci_pack, ci_pack*input->width(), ci_pack*input->width()*input->height()}},
      {(ElementInput_F16 *)mResource->mFilter, {ci_pack, ci_pack*mResource->mKernelInfo.kernelX, ci_pack*mResource->mKernelInfo.kernelX*mResource->mKernelInfo.kernelY}},
      {(ElementOutput_F16 *)mResource->mBias, {0, 0, 0}},
      {(ElementOutput_F16 *)output->deviceId(), {co_pack, co_pack*output->width(), co_pack*output->width()*output->height()}},
      {alpha, beta},
    };


    size_t workspace_size = mImplicitConvOp.get_workspace_size(arguments);

    if(workspace_size != 0) {
        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
        mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
        mWorkspace = (void *)workspaceTensor.get()->buffer().device;
    }

    // Check the problem size is supported or not 
    cutlass::Status status = mImplicitConvOp.can_implement(arguments);
    cutlass_check(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = mImplicitConvOp.initialize(arguments, (uint8_t *)mWorkspace);
    cutlass_check(status);

    return NO_ERROR;
}

ErrorCode ConvImplicitExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    auto status = mImplicitConvOp();
    cutlass_check(status);
    return NO_ERROR;
}


} // namespace CUDA
} // namespace MNN
