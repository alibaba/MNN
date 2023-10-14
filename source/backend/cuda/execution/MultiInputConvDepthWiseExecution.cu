//
//  MultiInputConvDepthWiseExecution.cpp
//  MNN
//
//  Created by MNN on 2023/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MultiInputConvDepthWiseExecution.hpp"

namespace MNN {
namespace CUDA {


template<typename type1, typename type2>
__global__ void WeightPrepare(const type1 * inputWeightDevice, type2 * outputWeightDevice, const int numTotal, const int numChannel, const int kernelHeight, const int kernelWeight, DivModFast divNumChannelPack, DivModFast divKernelWeight) {
    for (int indexOutput = blockDim.x * blockIdx.x + threadIdx.x; indexOutput < numTotal; indexOutput += blockDim.x * gridDim.x) {
        int indexChannel, tempOutputChannel, indexKernelWeight, indexKernelHeight;
        divNumChannelPack.divmod(indexOutput, tempOutputChannel, indexChannel);
        divKernelWeight.divmod(tempOutputChannel, indexKernelHeight, indexKernelWeight);

        if (indexChannel >= numChannel) {
            outputWeightDevice[indexOutput] = (type2)0.0f;
            continue;
        } else {
            int indexInput = (indexChannel * kernelHeight + indexKernelHeight) * kernelWeight + indexKernelWeight;
            outputWeightDevice[indexOutput] = (type2)inputWeightDevice[indexInput];
        }
    }

    return;
}

template<typename type1, typename type2>
__global__ void BiasPrepare(const type1 * inputBiasDevice, type2 * outputBiasDevice, const int numTotal, const int numChannel) {
    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < numTotal; index += blockDim.x * gridDim.x) {
        if (index >= numChannel)
        {
            outputBiasDevice[index] = (type2)0.0f;
            continue;
        }
        outputBiasDevice[index] = (type2)inputBiasDevice[index];
    }

    return;
}

template<typename type>
__global__ void BiasZeroPrepare(type * outputBiasDevice, const int numTotal) {
    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < numTotal; index += blockDim.x * gridDim.x) {
        outputBiasDevice[index] = (type)0.0f;
    }

    return;
}

MultiInputConvDepthWiseExecution::MultiInputConvDepthWiseExecution(const Op *op, Backend *bn) : Execution(bn) {
    mOp = op;
}

MultiInputConvDepthWiseExecution::~ MultiInputConvDepthWiseExecution() {
    //
}

ErrorCode MultiInputConvDepthWiseExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // prepare mParams from mOp and inputs[0]
    auto convCommon = mOp->main_as_Convolution2D()->common();
    auto pad = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], convCommon);
    mParams.inputSize[0] = inputs[0]->width();
    mParams.inputSize[1] = inputs[0]->height();
    mParams.outputSize[0] = outputs[0]->width();
    mParams.outputSize[1] = outputs[0]->height();
    mParams.kernelSize[0] = convCommon->kernelX();
    mParams.kernelSize[1] = convCommon->kernelY();
    mParams.stride[0] = convCommon->strideX();
    mParams.stride[1] = convCommon->strideY();
    mParams.pad[0] = pad.first;
    mParams.pad[1] = pad.second;
    mParams.dilate[0] = convCommon->dilateX();
    mParams.dilate[1] = convCommon->dilateY();
    mParams.channel_raw = inputs[0]->channel();
    mParams.channel_div = UP_DIV(inputs[0]->channel(), PACK_NUMBER);
    mParams.channel_pack = mParams.channel_div * PACK_NUMBER;
    mParams.batch = inputs[0]->batch();
    mParams.numWeightPackTotal = mParams.kernelSize[0] * mParams.kernelSize[1] * mParams.channel_pack;
    mParams.numBiasPackTotal = mParams.channel_pack;
    mParams.numOutputTotal = mParams.batch * mParams.outputSize[1] * mParams.outputSize[0] * mParams.channel_pack;
    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        // Do nothing
    } else {
        mParams.minValue = -FLT_MAX;
        mParams.maxValue = FLT_MAX;
    }
    if (convCommon->relu()) {
        mParams.minValue = 0.0f;
    }
    if (convCommon->relu6()) {
        mParams.minValue = 0.0f;
        mParams.maxValue = 6.0f;
    }

    // prepare mParams.mFilter and mParams.mBias
    auto pool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();

    auto bufferFilter = pool->alloc(mParams.numWeightPackTotal * sizeof(half));
    mParams.mFilter = (void*)((uint8_t*)bufferFilter.first + bufferFilter.second);

    auto bufferBias = pool->alloc(mParams.numBiasPackTotal * sizeof(half));
    mParams.mBias = (void*)((uint8_t*)bufferBias.first + bufferBias.second);

    pool->free(bufferFilter);
    pool->free(bufferBias);

    return NO_ERROR;
}

ErrorCode MultiInputConvDepthWiseExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto pool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
    auto& prop = runtime->prop();
    int limitThreads = UP_DIV(mParams.numOutputTotal, prop.multiProcessorCount);
    int threadNum = ALIMIN(prop.maxThreadsPerBlock/2, limitThreads);
    int blockNum = prop.multiProcessorCount;

    DivModFast d_oc(mParams.channel_div * PACK_NUMBER / 2);
    DivModFast d_ow(mParams.outputSize[0]);
    DivModFast d_oh(mParams.outputSize[1]);

    const int iw = mParams.inputSize[0];
    const int ih = mParams.inputSize[1];
    const int ow = mParams.outputSize[0];
    const int oh = mParams.outputSize[1];
    const int kw = mParams.kernelSize[0];
    const int kh = mParams.kernelSize[1];
    const int sw = mParams.stride[0];
    const int sh = mParams.stride[1];
    const int pw = mParams.pad[0];
    const int ph = mParams.pad[1];
    const int dw = mParams.dilate[0];
    const int dh = mParams.dilate[1];
    const int c_div = mParams.channel_div;
    const int c_p = mParams.channel_pack;
    const int channel_raw = mParams.channel_raw;
    const int batch = mParams.batch;
    const int numOutputTotal = mParams.numOutputTotal;

    const float maxV = mParams.maxValue;
    const float minV = mParams.minValue;

    // prepare mParams.mFilter and mParams.mBias
    DivModFast divNumChannelPack(mParams.channel_pack);
    DivModFast divKernelWeight(kw);

    int numThreadPrepare = runtime->threads_num();
    int numWeightPackTotal = mParams.numWeightPackTotal;
    int numBiasPackTotal = mParams.numBiasPackTotal;
    int numWeightBlock = UP_DIV(mParams.numWeightPackTotal, numThreadPrepare);
    int numBiasBlock = UP_DIV(mParams.numBiasPackTotal, numThreadPrepare);

    // prepare mParams.mFilter
    if (static_cast<CUDABackend *>(backend())->useFp16()) {
        WeightPrepare<<<numWeightBlock,numThreadPrepare>>>((const half *)inputs[1]->deviceId(), (half *)mParams.mFilter, numWeightPackTotal, channel_raw, kh, kw, divNumChannelPack, divKernelWeight);
    } else {
        WeightPrepare<<<numWeightBlock,numThreadPrepare>>>((const float *)inputs[1]->deviceId(), (half *)mParams.mFilter, numWeightPackTotal, channel_raw, kh, kw, divNumChannelPack, divKernelWeight);
    }

    // prepare mParams.mBias
    if(inputs.size() > 2) {
        if (static_cast<CUDABackend *>(backend())->useFp16()) {
            BiasPrepare<<<numBiasBlock, numThreadPrepare>>>((const half *)inputs[2]->deviceId(), (half *)mParams.mBias, numBiasPackTotal, channel_raw);
        } else {
            BiasPrepare<<<numBiasBlock, numThreadPrepare>>>((const float *)inputs[2]->deviceId(), (half *)mParams.mBias, numBiasPackTotal, channel_raw);
        }
    } else {
            BiasZeroPrepare<<<numBiasBlock, numThreadPrepare>>>((half *)mParams.mBias, numBiasPackTotal);
    }

    ErrorCode res = ConvDepthWiseCompute(backend(),
                                         blockNum,
                                         threadNum,
                                         (const void *)inputs[0]->deviceId(),
                                         mParams.mFilter,
                                         mParams.mBias,
                                         (void *)outputs[0]->deviceId(),
                                         maxV,
                                         minV,
                                         iw,
                                         ih,
                                         c_div,
                                         c_p,
                                         ow,
                                         oh,
                                         kw,
                                         kh,
                                         dw,
                                         dh,
                                         sw,
                                         sh,
                                         pw,
                                         ph,
                                         numOutputTotal,
                                         d_oc,
                                         d_ow,
                                         d_oh);

    return res;
}


}
}