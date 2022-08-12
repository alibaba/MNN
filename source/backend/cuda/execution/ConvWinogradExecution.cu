//
//  ConvWinogradExecution.cpp
//  MNN
//
//  Created by MNN on 2022/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "ConvWinogradExecution.hpp"
#include "math/WingoradGenerater.hpp"
#include "WinogradTrans.cuh"

namespace MNN {
namespace CUDA {

#define UNIT 2
__global__ void WinoWeightReorder(const float* GgGt, 
    half* GgGt_trans,
    const int outside,
    const int unitCi,
    const int unitCo
    ) {
    const int maxCount = outside * unitCi * unitCo;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += gridDim.x * blockDim.x) {
        size_t outside_idx =  index / (unitCi*unitCo);
        size_t tmp =  index % (unitCi*unitCo);
        size_t ci_idx = tmp / unitCo;
        size_t co_idx = tmp % unitCo;

        // [4x4, Cop, Cip, unitCi, unitCo] -->> [4x4, Cop, Cip, unitCo, unitCi]
        size_t dst_idx = outside_idx * (unitCi*unitCo) + co_idx * unitCi + ci_idx;
        *(GgGt_trans + dst_idx) = *(GgGt + index);
    }
}

bool ConvWinogradExecution::isValid(const Convolution2D* conv, const Tensor* input) {
    //return false;
    if(conv->common()->strideX() != 1 || conv->common()->strideY() != 1) {
        return false;
    }
    if(conv->common()->dilateX() != 1 || conv->common()->dilateY() != 1) {
        return false;
    }
    if(conv->common()->padX() != 1 || conv->common()->padY() != 1) {
        return false;
    }
    return (conv->common()->kernelX() == 3) && (conv->common()->kernelY() == 3);
}

ConvWinogradExecution::Resource::Resource(Backend* backend, const MNN::Op* op) {
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
    mKernelInfo.activationType = common->relu() ? 1 : (common->relu6() ? 2 : 0);

    //weight host->device
    const float* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, conv, &filterDataPtr, &weightSize);
    mKernelInfo.kernelN = common->outputCount();
    mKernelInfo.kernelC = weightSize / mKernelInfo.kernelN / mKernelInfo.kernelX / mKernelInfo.kernelY;

    const int kernel = 3;
    Math::WinogradGenerater generator(UNIT, kernel, 1.0);
    std::shared_ptr<Tensor> srcWeight(Tensor::create<float>({mKernelInfo.kernelN, mKernelInfo.kernelC, mKernelInfo.kernelY, mKernelInfo.kernelX},
        (void *)filterDataPtr, Tensor::CAFFE));

    auto dstWeight = generator.allocTransformWeight(srcWeight.get(), MATMULPACK, MATMULPACK);
    generator.transformWeight(dstWeight.get(), srcWeight.get());
    auto dstWeightSize = dstWeight->elementSize();

    // Reorder weight
    {
        auto tempCacheBuffer = static_cast<CUDABackend*>(backend)->getStaticBufferPool()->alloc(dstWeightSize*sizeof(float));
        float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
        runtime->memcpy(cacheWeight, dstWeight->host<uint8_t>(), dstWeightSize * sizeof(float), MNNMemcpyHostToDevice);
        weightTensor.reset(Tensor::createDevice<int16_t>({dstWeightSize}));
        backend->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
        mFilter = (void *)weightTensor.get()->buffer().device;
        auto& prop = runtime->prop();
        int cores = prop.multiProcessorCount;
        int threadNumbers = prop.maxThreadsPerBlock;

        int coDiv = UP_DIV(mKernelInfo.kernelN, MATMULPACK);
        int ciDiv = UP_DIV(mKernelInfo.kernelC, MATMULPACK);

        WinoWeightReorder<<<cores, threadNumbers>>>((float*)cacheWeight, (half*)mFilter,
                (UNIT+kernel-1) * (UNIT+kernel-1) * coDiv * ciDiv, MATMULPACK, MATMULPACK);

        static_cast<CUDABackend*>(backend)->getStaticBufferPool()->free(tempCacheBuffer);
    }
    
    // Copy Bias
    int biasSize = conv->bias()->size();
    int alignSize = UP_DIV(biasSize, MATMULPACK) * MATMULPACK;
    biasTensor.reset(Tensor::createDevice<uint32_t>({alignSize}));
    backend->onAcquireBuffer(biasTensor.get(), Backend::STATIC);

    mBias = (void *)biasTensor.get()->buffer().device;
    cuda_check(cudaMemset(mBias, 0, alignSize*sizeof(float)));
    cuda_check(cudaMemcpy(mBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));

}

ConvWinogradExecution::Resource::~Resource() {
    // Do nothing
}

ConvWinogradExecution::ConvWinogradExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res)  : Execution(backend), mOp(op) {
    mResource = res;
    auto staticPool = static_cast<CUDABackend*>(backend)->getStaticBufferPool();
    mGpuMatMulParam = staticPool->alloc(sizeof(MatMulParam));
}
ConvWinogradExecution::~ConvWinogradExecution() {
    auto staticPool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
    staticPool->free(mGpuMatMulParam);
}

bool ConvWinogradExecution::onClone(Backend* backend, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if(nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvWinogradExecution(backend, op, mResource);
    *dst = dstExe;
    return true;
}

ErrorCode ConvWinogradExecution::onResize(const std::vector<Tensor*>  &inputs, const std::vector<Tensor*> &outputs) {

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    auto input = inputs[0];
    auto output = outputs[0];
    auto convCommon = mOp->main_as_Convolution2D()->common();
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mOp->main_as_Convolution2D()->common());
    mPadX = std::get<0>(pads);
    mPadY = std::get<1>(pads);
    int ic = input->channel();
    int icDiv = UP_DIV(ic, MATMULPACK);
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(input);

    auto wUnit = UP_DIV(output->width(), UNIT);
    auto hUnit = UP_DIV(output->height(), UNIT);

    int e = wUnit * hUnit * output->batch();
    int l = icDiv * MATMULPACK;
    int h = output->channel();
    mMatMulParam.elh[0] = e;
    mMatMulParam.elh[1] = l;
    mMatMulParam.elh[2] = h;

    int ePack = MATMULPACK;
    int hPack = MATMULPACK;
    mMatMulParam.elhPack[0] = UP_DIV(e, ePack);
    mMatMulParam.elhPack[1] = UP_DIV(l, MATMULPACK);
    mMatMulParam.elhPack[2] = UP_DIV(h, hPack);
    // mMatMulParam.cStride[0] = mIm2ColParamter.ow * mIm2ColParamter.oh * h;
    // mMatMulParam.cStride[1] = 1;
    // mMatMulParam.cStride[2] = mIm2ColParamter.ow * mIm2ColParamter.oh;
    mMatMulParam.minValue = -FLT_MAX;
    mMatMulParam.maxValue = FLT_MAX;
    if (convCommon->relu()) {
        mMatMulParam.minValue = 0.0f;
    }
    if (convCommon->relu6()) {
        mMatMulParam.minValue = 0.0f;
        mMatMulParam.maxValue = 6.0f;
    }
    //MNN_PRINT("!!conv size:3-1, %d-%d-%d, %d-%d-%d\n", input->height(), input->width(), input->channel(), output->height(), output->width(), output->channel());

    runtime->memcpy((uint8_t*)mGpuMatMulParam.first + mGpuMatMulParam.second, &mMatMulParam, sizeof(MatMulParam), MNNMemcpyHostToDevice);

    int block = UNIT + convCommon->kernelY() - 1;
    mBlock2 = block * block;
    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    auto buffer = pool->alloc((size_t)sizeof(__half) * mBlock2 * mMatMulParam.elhPack[0] * mMatMulParam.elhPack[1] * (size_t)ePack * (size_t)MATMULPACK);
    mBtdB_Buffer = (__half*)((uint8_t*)buffer.first + buffer.second);
    
    auto buffer2 = pool->alloc(bytes * mBlock2 * mMatMulParam.elh[0] * mMatMulParam.elhPack[2] * (size_t)hPack);
    mMatmul_Buffer = (void*)((uint8_t*)buffer2.first + buffer2.second);
    
    pool->free(buffer);
    pool->free(buffer2);
    return NO_ERROR;
}

ErrorCode ConvWinogradExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input = inputs[0];
    auto output = outputs[0];
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    int threadNumbers = prop.maxThreadsPerBlock / 2;
    auto gpuMatMul = (const MatMulParam*)((uint8_t*)mGpuMatMulParam.first + mGpuMatMulParam.second);

    int coDiv = UP_DIV(mResource->mKernelInfo.kernelN, MATMULPACK);
    int ciDiv = UP_DIV(mResource->mKernelInfo.kernelC, MATMULPACK);

    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(input);
    const void *input_addr = (const void*)input->deviceId();
    const void *mGgGt_Buffer = mResource->mFilter;
    const void *bias_addr = mResource->mBias;
    void *output_addr = (void*)output->deviceId();

    const int kernel = 3;
    if(bytes == 4) {
        WinoInputTrans<<<cores, threadNumbers>>>((const float*)input_addr, (half*)mBtdB_Buffer, UNIT,
                (UNIT+kernel-1)*(UNIT+kernel-1), input->channel(), ciDiv, output->batch(), UP_DIV(input->width(), UNIT), 
                UP_DIV(input->height(), UNIT), MATMULPACK, MATMULPACK,
                mPadX, mPadY, input->width(), input->height());
        checkKernelErrors;
    } else {
        WinoInputTrans<<<cores, threadNumbers>>>((const half*)input_addr, (half*)mBtdB_Buffer, UNIT,
                (UNIT+kernel-1)*(UNIT+kernel-1), input->channel(), ciDiv, output->batch(), UP_DIV(input->width(), UNIT), 
                UP_DIV(input->height(), UNIT), MATMULPACK, MATMULPACK,
                mPadX, mPadY, input->width(), input->height());
        checkKernelErrors;
    }

    int maxThreadInWarp = UP_DIV(mBlock2 * mMatMulParam.elhPack[0] * mMatMulParam.elhPack[2], cores);
    int threads_num = std::min(prop.maxThreadsPerBlock/2, maxThreadInWarp * prop.warpSize);
    int basicMemory = 16 * 16 * sizeof(float) * prop.maxThreadsPerBlock / prop.warpSize;

    int iBlock = 0;
    if (4 == bytes) {
        cudaFuncSetAttribute(GemmPackedMulti<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, prop.sharedMemPerMultiprocessor);
        GemmPackedMulti<<<cores, threads_num, basicMemory>>>(gpuMatMul, iBlock, mBlock2, (float*)mMatmul_Buffer, mBtdB_Buffer, (const __half*)mGgGt_Buffer);
        checkKernelErrors;
    } else {
        //MNN_PRINT("%d - %d, %d- %d\n", cpuParam->elhPack[0], cpuParam->elhPack[2], cpuParam->elh[0], cpuParam->elh[2]);
        cudaFuncSetAttribute(GemmPackedMulti<half>, cudaFuncAttributeMaxDynamicSharedMemorySize, prop.sharedMemPerMultiprocessor);
        GemmPackedMulti<<<cores, threads_num, basicMemory>>>(gpuMatMul, iBlock, mBlock2, (half*)mMatmul_Buffer, mBtdB_Buffer, (const __half*)mGgGt_Buffer);
        checkKernelErrors;
    }

    if (4 == bytes) {
        WinoTrans2Output<<<cores, threadNumbers>>>((const float*)mMatmul_Buffer, (const float*)bias_addr, (float*)output_addr,
                gpuMatMul, UNIT,
                mBlock2, output->channel(), ciDiv, output->batch(), UP_DIV(output->width(), UNIT), 
                UP_DIV(output->height(), UNIT), MATMULPACK, MATMULPACK,
                output->width(), output->height());
        checkKernelErrors;
    } else {
        WinoTrans2Output<<<cores, threadNumbers>>>((const half*)mMatmul_Buffer, (const float*)bias_addr, (half*)output_addr,
                gpuMatMul, UNIT,
                mBlock2, output->channel(), ciDiv, output->batch(), UP_DIV(output->width(), UNIT), 
                UP_DIV(output->height(), UNIT), MATMULPACK, MATMULPACK,
                output->width(), output->height());
        checkKernelErrors;
    }


    // if(output->width() == 56 && output->channel() == 64 && input->channel() == 64) {
    //     cudaDeviceSynchronize();
    //     float bias_[mMatMulParam.elhPack[2] * 16];
    //     runtime->memcpy((void*)bias_, bias_addr, mMatMulParam.elhPack[2] * 16*sizeof(float), MNNMemcpyDeviceToHost);
    //     for(int i=0; i<mMatMulParam.elhPack[2] * 16; i++) {
    //         printf("%d-%f\n", i, bias_[i]);
    //     }
    // }
    return NO_ERROR;
}


} // namespace CUDA
} // namespace MNN