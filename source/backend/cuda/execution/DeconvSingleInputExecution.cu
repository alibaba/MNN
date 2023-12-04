//
//  DeconvSingleInputExecution.cpp
//  MNN
//
//  Created by MNN on 2022/03/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "DeconvSingleInputExecution.hpp"
#include "MultiInputDeconvExecution.hpp"
#include "ConvBaseKernel.cuh"
#include "DeconvBaseKernel.cuh"

namespace MNN {
namespace CUDA {

DeconvSingleInputExecution::Resource::Resource(Backend* bn, const MNN::Op* op) {
    mBackend = bn;
    auto runtime = static_cast<CUDABackend*>(bn)->getCUDARuntime();
    
    auto conv       = op->main_as_Convolution2D();
    auto common     = conv->common();
    mKernelInfo.kernelX        = common->kernelX();
    mKernelInfo.kernelY        = common->kernelY();
    mKernelInfo.groups         = common->group();
    mKernelInfo.strideX        = common->strideX();
    mKernelInfo.strideY        = common->strideY();
    mKernelInfo.dilateX        = common->dilateX();
    mKernelInfo.dilateY        = common->dilateY();
    mKernelInfo.activationType = common->relu() ? 1 : (common->relu6() ? 2 : 0);

    //weight host->device
    const float* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, bn, conv, &filterDataPtr, &weightSize);
    mKernelInfo.kernelN = common->outputCount();
    mKernelInfo.kernelC = weightSize / mKernelInfo.kernelN / mKernelInfo.kernelX / mKernelInfo.kernelY;

    CutlassGemmInfo param;
    int l = UP_DIV(mKernelInfo.kernelC, PACK_NUMBER) * PACK_NUMBER;
    int h = mKernelInfo.kernelN * mKernelInfo.kernelX * mKernelInfo.kernelY;

    param.elh[1] = l;
    param.elh[2] = h;
    param.elhPad[1] = UP_DIV(l, PACK_NUMBER) * PACK_NUMBER;
    param.elhPad[2] = UP_DIV(h, PACK_NUMBER) * PACK_NUMBER;

    auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(float));
    float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
    runtime->memcpy(cacheWeight, filterDataPtr, weightSize * sizeof(float), MNNMemcpyHostToDevice);
    
    // Reorder weight
    if(static_cast<CUDABackend*>(bn)->getPrecision() == 1) {
        weightTensor.reset(Tensor::createDevice<int32_t>({param.elhPad[1] * param.elh[2]}));
    } else {
        weightTensor.reset(Tensor::createDevice<int16_t>({param.elhPad[1] * param.elh[2]}));
    }
    bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
    mFilter = (void *)weightTensor.get()->buffer().device;    
    
    callWeightReorder((const void *)cacheWeight, (void *)mFilter, mKernelInfo, param.elhPad[1], (int)(static_cast<CUDABackend*>(bn)->getPrecision() == 1), runtime);

    static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempCacheBuffer);

    // Copy Bias
    int biasSize = conv->bias()->size();
    if(static_cast<CUDABackend*>(bn)->getPrecision() == 2) {
        // Pack for flaoot22half2 memory protect
        int biasPackSize = UP_DIV(biasSize, PACK_NUMBER) * PACK_NUMBER;

        biasTensor.reset(Tensor::createDevice<float>({biasPackSize}));
        bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
        mBias = (void *)biasTensor.get()->buffer().device;

        auto tempBiasBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(biasPackSize * sizeof(float));
        float* cacheBias = (float*)((uint8_t*)tempBiasBuffer.first + tempBiasBuffer.second);
        cuda_check(cudaMemcpy(cacheBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));

        callFloat2Half((const void*)cacheBias, (void*)mBias, biasPackSize, runtime);
    } else {
        biasTensor.reset(Tensor::createDevice<float>({biasSize}));
        bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
        mBias = (void *)biasTensor.get()->buffer().device;
        cuda_check(cudaMemcpy(mBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));
    }
}

DeconvSingleInputExecution::Resource::~Resource() {
    // Do nothing
}
DeconvSingleInputExecution::DeconvSingleInputExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res) : CutlassDeconvCommonExecution(backend) {
    mResource = res;
    mOp = op;
    mPrecisonLevel = static_cast<CUDABackend*>(backend)->getPrecision();
    mFp16Infer = (mPrecisonLevel == 2);
    mFp32Infer = (mPrecisonLevel == 1);
    mFp16Fp32MixInfer = (mPrecisonLevel == 0);
}

DeconvSingleInputExecution::~DeconvSingleInputExecution() {
    // Do nothing
}
bool DeconvSingleInputExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new DeconvSingleInputExecution(bn, op, mResource);
    *dst = dstExe;
    return true;
}


ErrorCode DeconvSingleInputExecution::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input = inputs[0], output = outputs[0];
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);
    auto convCommon = mOp->main_as_Convolution2D()->common();

    // Col2Im Param
    auto pad = ConvolutionCommon::convolutionTransposePad(input, output, mOp->main_as_Convolution2D()->common());
    mCol2ImParamter.dilateX         = convCommon->dilateX();
    mCol2ImParamter.dilateY         = convCommon->dilateY();
    mCol2ImParamter.strideX         = convCommon->strideX();
    mCol2ImParamter.strideY         = convCommon->strideY();
    mCol2ImParamter.ic              = input->channel();
    mCol2ImParamter.oc              = output->channel();
    mCol2ImParamter.kernelX         = convCommon->kernelX();
    mCol2ImParamter.kernelY         = convCommon->kernelY();
    mCol2ImParamter.padX = pad.first;
    mCol2ImParamter.padY = pad.second;

    mCol2ImParamter.ih = input->height();
    mCol2ImParamter.iw = input->width();
    mCol2ImParamter.oh = output->height();
    mCol2ImParamter.ow = output->width();
    mCol2ImParamter.ob = output->batch();

    mActivationType = convCommon->relu() ? 1 : convCommon->relu6() ? 2 : 0;

    // Matmul Param
    int e = input->height() * input->width() * output->batch();
    int l = UP_DIV(input->channel(), PACK_NUMBER) * PACK_NUMBER;
    int h = output->channel() * mCol2ImParamter.kernelX * mCol2ImParamter.kernelY;

    mGemmInfo.elh[0] = e;
    mGemmInfo.elh[1] = l;
    mGemmInfo.elh[2] = h;
    mGemmInfo.elhPad[0] = UP_DIV(e, PACK_NUMBER) * PACK_NUMBER;
    mGemmInfo.elhPad[1] = UP_DIV(l, PACK_NUMBER) * PACK_NUMBER;
    mGemmInfo.elhPad[2] = UP_DIV(h, PACK_NUMBER) * PACK_NUMBER;


    // Alloc temp cuda memory
    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    MemChunk buffer_input, buffer_im2col;
    if(mFp16Fp32MixInfer) {
        buffer_input = pool->alloc(sizeof(__half) * mGemmInfo.elh[0] * mGemmInfo.elhPad[1]);
        mInputBuffer = (void*)((uint8_t*)buffer_input.first + buffer_input.second);
    } else {
        mInputBuffer = (void*)input->deviceId();
    }
    buffer_im2col = pool->alloc(bytes * mGemmInfo.elh[0] * mGemmInfo.elhPad[2]);
    mIm2ColBuffer = (__half*)((uint8_t*)buffer_im2col.first + buffer_im2col.second);
    if(mFp16Fp32MixInfer) {
        pool->free(buffer_input);
    }
    pool->free(buffer_im2col);

    if(mFp16Fp32MixInfer || mFp32Infer) {
        mZeroTensor.reset(Tensor::createDevice<uint32_t>({mGemmInfo.elhPad[2]}));
    } else {
        mZeroTensor.reset(Tensor::createDevice<uint16_t>({mGemmInfo.elhPad[2]}));
    }
    static_cast<CUDABackend*>(backend())->onAcquireBuffer(mZeroTensor.get(), Backend::STATIC);

    mZeroPtr = (void *)mZeroTensor.get()->buffer().device;
    cuda_check(cudaMemset(mZeroPtr, 0, mGemmInfo.elhPad[2]*bytes));


    mFilterAddr = mResource->mFilter;
    mBiasAddr   = mResource->mBias;
    mBackendPtr = mResource->mBackend;
 
    // Call from different function
    if(mFp32Infer){
        return callCutlassGemmCudaCoreFloat32(inputs, outputs);
    } 
 
    mGpuComputeCap = runtime->compute_capability();
    //MNN_PRINT("Gpu smArch is sm_%d\n", mGpuComputeCap);
    if(mGpuComputeCap < 75) {
        return callCutlassGemmCudaCoreFloat16(inputs, outputs);
    }
    return callCutlassGemmTensorCore(inputs, outputs);

}

ErrorCode DeconvSingleInputExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    //MNN_PRINT("cuda convSingleInput onExecute in, inputsize:%d %d\n", (int)inputs.size(), workspace_size_);
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    const void *input_addr = (const void*)inputs[0]->deviceId();
    const void *filter_addr = mResource->mFilter;
    const void *bias_addr = mResource->mBias;
    void *output_addr = (void*)outputs[0]->deviceId();

    // Do input Rerange Pack
    if(mFp16Fp32MixInfer) {
        size_t maxCount = mGemmInfo.elh[0] * mGemmInfo.elhPad[1];
        callFloat2Half((const void*)input_addr, (void*)mInputBuffer, maxCount, runtime);
    } 

    // Run cutlass gemm forward
    runCutlassGemmFunc();

    // Run Col2Im
    int convert_flag = mPrecisonLevel;
    if(convert_flag == 0) {
        convert_flag = 1;
    }
    callCol2ImFunc((const void*)mIm2ColBuffer, (const void*)bias_addr, (void *)output_addr, &mCol2ImParamter, convert_flag, runtime);

    return NO_ERROR;
}

class CUDADeconvolutionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, 
            const MNN::Op* op, Backend* backend) const override {
        if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
            auto quan = op->main_as_Convolution2D()->quanParameter();
            if (1 == quan->type() || 2 == quan->type()) {
                MNN_PRINT("cuda Deconv quant type 1 or 2 not support\n");
                return nullptr;
            }
        }

        if(inputs.size() == 2 || inputs.size() == 3) {
            return new MultiInputDeconvExecution(op, backend);
        } else if(inputs.size() == 1) {
            std::shared_ptr<DeconvSingleInputExecution::Resource> resource(new DeconvSingleInputExecution::Resource(backend, op));
            return new DeconvSingleInputExecution(backend, op, resource);
        } else {
            MNN_PRINT("Deconv inputs size:%d not support", (int)inputs.size());
            return nullptr;
        }
    }
};

CUDACreatorRegister<CUDADeconvolutionCreator> __DeConvExecution(OpType_Deconvolution);

}// namespace CUDA
}// namespace MNN
