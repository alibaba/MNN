//
//  CoreMLPool.cpp
//  MNN
//
//  Created by MNN on 2021/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLPool.hpp"

namespace MNN {


CoreMLPool::CoreMLPool(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
    initLayer();
}

void CoreMLPool::addPadLayer(const Tensor * input, const Pool* common) {
    MNN_ASSERT(common->padType() == PoolPadType_CAFFE);
    int top, left, bottom, right;
    if (nullptr != common->pads()) {
        MNN_ASSERT(common->pads()->size() >= 4);
        top = common->pads()->Get(0);
        left = common->pads()->Get(1);
        bottom = common->pads()->Get(2);
        right = common->pads()->Get(3);
    } else {
        top = common->padY();
        left = common->padX();
        bottom = common->padY();
        right = common->padX();
    }
    if (top == 0 && left == 0 && bottom == 0 && right == 0) {
        return;
    }
    auto paddingLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
    core_ml__specification__neural_network_layer__init(paddingLayer);
    paddingLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_PADDING;
    mCoreMLBackend->setLayerName(paddingLayer, "PoolPadding");
    paddingLayer->padding = mCoreMLBackend->create<CoreML__Specification__PaddingLayerParams>();
    core_ml__specification__padding_layer_params__init(paddingLayer->padding);
    paddingLayer->padding->padding_type_case = CORE_ML__SPECIFICATION__PADDING_LAYER_PARAMS__PADDING_TYPE_CONSTANT;
    paddingLayer->padding->constant = mCoreMLBackend->create<CoreML__Specification__PaddingLayerParams__PaddingConstant>();
    core_ml__specification__padding_layer_params__padding_constant__init(paddingLayer->padding->constant);
    paddingLayer->padding->constant->value = 0;
    paddingLayer->padding->paddingamounts = mCoreMLBackend->create<CoreML__Specification__BorderAmounts>();
    core_ml__specification__border_amounts__init(paddingLayer->padding->paddingamounts);
    paddingLayer->padding->paddingamounts->n_borderamounts = 2;
    paddingLayer->padding->paddingamounts->borderamounts = mCoreMLBackend->create<CoreML__Specification__BorderAmounts__EdgeSizes*>(2);
    paddingLayer->padding->paddingamounts->borderamounts[0] = mCoreMLBackend->create<CoreML__Specification__BorderAmounts__EdgeSizes>();
    core_ml__specification__border_amounts__edge_sizes__init(paddingLayer->padding->paddingamounts->borderamounts[0]);
    paddingLayer->padding->paddingamounts->borderamounts[0]->startedgesize = top;
    paddingLayer->padding->paddingamounts->borderamounts[0]->endedgesize = bottom;
    paddingLayer->padding->paddingamounts->borderamounts[1] = mCoreMLBackend->create<CoreML__Specification__BorderAmounts__EdgeSizes>();
    core_ml__specification__border_amounts__edge_sizes__init(paddingLayer->padding->paddingamounts->borderamounts[1]);
    paddingLayer->padding->paddingamounts->borderamounts[1]->startedgesize = left;
    paddingLayer->padding->paddingamounts->borderamounts[1]->endedgesize = right;
    auto inputName = mPoolInputName;
    mPoolInputName = mPoolInputName + "-" + mPoolOutputName + "-Padding";
    setLayerInputsAndOutputs(paddingLayer, {inputName}, {mPoolInputName});
    mCoreMLBackend->addLayer(paddingLayer);
}

ErrorCode CoreMLPool::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    mPoolInputName = mCoreMLBackend->getTensorName(inputs[0]);
    mPoolOutputName = mCoreMLBackend->getTensorName(outputs[0]);
    auto pool    = mOp->main_as_Pool();
    auto strideX = pool->strideX();
    auto strideY = pool->strideY();
    auto kernelX = pool->kernelX();
    auto kernelY = pool->kernelY();
    auto padMod  = pool->padType();
    auto global  = pool->isGlobal();
    mLayer_->pooling = mCoreMLBackend->create<CoreML__Specification__PoolingLayerParams>();
    mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_POOLING;
    core_ml__specification__pooling_layer_params__init(mLayer_->pooling);
    mLayer_->pooling->globalpooling = global;
    mLayer_->pooling->n_stride = 2;
    mLayer_->pooling->stride = mCoreMLBackend->create<uint64_t>(mLayer_->pooling->n_stride);
    mLayer_->pooling->stride[0] = strideY;
    mLayer_->pooling->stride[1] = strideX;
    mLayer_->pooling->n_kernelsize = 2;
    mLayer_->pooling->kernelsize = mCoreMLBackend->create<uint64_t>(mLayer_->pooling->n_kernelsize);
    mLayer_->pooling->kernelsize[0] = kernelY;
    mLayer_->pooling->kernelsize[1] = kernelX;
    switch (padMod) {
        case PoolPadType_SAME:
            mLayer_->pooling->pooling_padding_type_case = CORE_ML__SPECIFICATION__POOLING_LAYER_PARAMS__POOLING_PADDING_TYPE_SAME;
            mLayer_->pooling->same = mCoreMLBackend->create<CoreML__Specification__SamePadding>();
            core_ml__specification__same_padding__init(mLayer_->pooling->same);
            break;
        case PoolPadType_VALID:
            mLayer_->pooling->pooling_padding_type_case = CORE_ML__SPECIFICATION__POOLING_LAYER_PARAMS__POOLING_PADDING_TYPE_VALID;
            mLayer_->pooling->valid = mCoreMLBackend->create<CoreML__Specification__ValidPadding>();
            core_ml__specification__valid_padding__init(mLayer_->pooling->valid);
            break;
        case PoolPadType_CAFFE:
            if ((pool->pads() && pool->pads()->size() > 0) || pool->padX() > 0) {
                addPadLayer(inputs[0], pool);
                mLayer_->pooling->pooling_padding_type_case = CORE_ML__SPECIFICATION__POOLING_LAYER_PARAMS__POOLING_PADDING_TYPE_VALID;
                mLayer_->pooling->valid = mCoreMLBackend->create<CoreML__Specification__ValidPadding>();
                core_ml__specification__valid_padding__init(mLayer_->pooling->valid);
            } else {
                mLayer_->pooling->pooling_padding_type_case = CORE_ML__SPECIFICATION__POOLING_LAYER_PARAMS__POOLING_PADDING_TYPE_INCLUDE_LAST_PIXEL;
                mLayer_->pooling->includelastpixel = mCoreMLBackend->create<CoreML__Specification__PoolingLayerParams__ValidCompletePadding>();
                core_ml__specification__pooling_layer_params__valid_complete_padding__init(mLayer_->pooling->includelastpixel);
            }
            break;
        default:
            break;
    }
    if (pool->type() == PoolType_AVEPOOL) {
        mLayer_->pooling->type = CORE_ML__SPECIFICATION__POOLING_LAYER_PARAMS__POOLING_TYPE__AVERAGE;
        mLayer_->pooling->avgpoolexcludepadding = true;
    } else {
        mLayer_->pooling->type = CORE_ML__SPECIFICATION__POOLING_LAYER_PARAMS__POOLING_TYPE__MAX;
    }
    setLayerInputsAndOutputs(mLayer_, {mPoolInputName}, {mPoolOutputName});
    mCoreMLBackend->addLayer(mLayer_);
    return NO_ERROR;
}


REGISTER_COREML_OP_CREATOR(CoreMLPool, OpType_Pooling)
} // namespace MNN
