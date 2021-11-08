//
//  TRTPooling.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTPooling.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"

using namespace std;

namespace MNN {

TRTPooling::TRTPooling(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                       const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
}

std::vector<ITensor *> TRTPooling::onEncode(const std::vector<ITensor *> &xOp) {
    auto input     = mInputs[0];
    auto output    = mOutputs[0];
    auto poolParam = mOp->main_as_Pool();

    nvinfer1::DimsHW nvKsize            = {poolParam->kernelY(), poolParam->kernelX()};
    nvinfer1::DimsHW nvStrides          = {poolParam->strideY(), poolParam->strideX()};
    nvinfer1::DimsHW nvPad              = {poolParam->padY(), poolParam->padX()};
    nvinfer1::IPoolingLayer *pool_layer = nullptr;

    if (poolParam->isGlobal()) {
        nvKsize   = {mInputs[0]->height(), mInputs[0]->width()};
        nvStrides = {mInputs[0]->height(), mInputs[0]->width()};
    }

    if (PoolType_MAXPOOL == poolParam->type()) {
        pool_layer = mTrtBackend->getNetwork()->addPooling(*(xOp[0]), PoolingType::kMAX, nvKsize);
    } else if (PoolType_AVEPOOL == poolParam->type()) {
        pool_layer = mTrtBackend->getNetwork()->addPooling(*(xOp[0]), PoolingType::kAVERAGE, nvKsize);
    } else {
        MNN_ERROR("poolint not support this type !!!\n");
    }

#ifdef TRT_LOG
    printf("pool pad mode TODO!\n");
    printf("pool size:%d %d %d, global:%d\n", outputs[0]->channel(), outputs[0]->height(), outputs[0]->width(),
           poolParam->isGlobal());
#endif

    pool_layer->setStride(nvStrides);
    if (poolParam->padType() == PoolPadType_SAME) {
        pool_layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    } else if (poolParam->padType() == PoolPadType_CAFFE) {
        pool_layer->setPaddingMode(nvinfer1::PaddingMode::kCAFFE_ROUND_DOWN);
    }
    pool_layer->setPadding(nvPad);
    return {pool_layer->getOutput(0)};
}

TRTCreatorRegister<TypedCreator<TRTPooling>> __pooling_op(OpType_Pooling);

} // namespace MNN
