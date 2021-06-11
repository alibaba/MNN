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

ErrorCode CoreMLPool::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
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
            // TODO: deal caffe pad mode
            mLayer_->pooling->pooling_padding_type_case = CORE_ML__SPECIFICATION__POOLING_LAYER_PARAMS__POOLING_PADDING_TYPE_INCLUDE_LAST_PIXEL;
            mLayer_->pooling->includelastpixel = mCoreMLBackend->create<CoreML__Specification__PoolingLayerParams__ValidCompletePadding>();
            core_ml__specification__pooling_layer_params__valid_complete_padding__init(mLayer_->pooling->includelastpixel);
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
    setLayerInputsAndOutputs(mLayer_, {mCoreMLBackend->getTensorName(inputs[0])}, {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->addLayer(mLayer_);
    return NO_ERROR;
}


CoreMLCreatorRegister<TypedCreator<CoreMLPool>> __pool_op(OpType_Pooling);


} // namespace MNN
