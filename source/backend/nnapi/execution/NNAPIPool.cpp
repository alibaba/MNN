//
//  NNAPIPool.cpp
//  MNN
//
//  Created by MNN on 2022/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPIPool.hpp"

namespace MNN {


NNAPIPool::NNAPIPool(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NNAPICommonExecution(b, op) {
}

ErrorCode NNAPIPool::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto pool      = mOp->main_as_Pool();
    auto strideX   = pool->strideX();
    auto strideY   = pool->strideY();
    auto kernelX   = pool->kernelX();
    auto kernelY   = pool->kernelY();
    auto padMod    = pool->padType();
    auto global    = pool->isGlobal();
    auto ceilModel = pool->ceilModel();
    int top, left, bottom, right;
    if (nullptr != pool->pads()) {
        MNN_ASSERT(pool->pads()->size() >= 4);
        top = pool->pads()->Get(0);
        left = pool->pads()->Get(1);
        bottom = pool->pads()->Get(2);
        right = pool->pads()->Get(3);
    } else {
        top = pool->padY();
        left = pool->padX();
        bottom = pool->padY();
        right = pool->padX();
    }
    if (padMod == PoolPadType_SAME || (ceilModel && (top + bottom + left + right) == 0)) {
        int inputY = (outputs[0]->height() - 1) * strideY + (kernelY - 1) + 1;
        int inputX = (outputs[0]->width() - 1) * strideX + (kernelX - 1) + 1;
        int padY = std::max(inputY - inputs[0]->height(), 0);
        int padX = std::max(inputY - inputs[0]->width(), 0);
        top = bottom = padY / 2;
        left = right = padX / 2;
        top += padY % 2;
        left += padX % 2;
    }
    if (global) {
        strideX = 1;
        strideY = 1;
        kernelX = inputs[0]->width();
        kernelY = inputs[0]->height();
    }
    // NNAPI Pool inputs: [input, pad_left, pad_right, pad_top, pad_bottom, stride_w, stride_h, kernel_w, kernel_h, fusecode, NCHW/NHWC]
    auto inputIdxs = getTensorIdxs(inputs);
    // pad
    inputIdxs.push_back(buildScalar(left));
    inputIdxs.push_back(buildScalar(right));
    inputIdxs.push_back(buildScalar(top));
    inputIdxs.push_back(buildScalar(bottom));
    // stride
    inputIdxs.push_back(buildScalar(strideX));
    inputIdxs.push_back(buildScalar(strideY));
    // kernel
    inputIdxs.push_back(buildScalar(kernelX));
    inputIdxs.push_back(buildScalar(kernelY));
    // fusecode
    inputIdxs.push_back(buildScalar(ANEURALNETWORKS_FUSED_NONE));
    // NCHW/NHWC
    inputIdxs.push_back(buildScalar(mNCHW));
    auto op = ANEURALNETWORKS_MAX_POOL_2D;
    if (pool->type() == PoolType_AVEPOOL) {
        op = ANEURALNETWORKS_AVERAGE_POOL_2D;
    }
    return buildOperation(op, inputIdxs, getTensorIdxs(outputs));
}

REGISTER_NNAPI_OP_CREATOR(NNAPIPool, OpType_Pooling)
} // namespace MNN
