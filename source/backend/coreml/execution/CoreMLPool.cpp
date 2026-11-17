//
//  CoreMLPool.cpp
//  MNN
//
//  Created by MNN on 2021/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLPool.hpp"

namespace MNN {

CoreMLPool::CoreMLPool(MNN::Backend* b, const MNN::Op* op, const std::vector<Tensor*>& inputs,
                       const std::vector<MNN::Tensor*>& outputs)
    : CoreMLCommonExecution(b, op) {}

ErrorCode CoreMLPool::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    auto pool = mOp->main_as_Pool();
    auto input = inputs[0];
    auto output = outputs[0];
    bool average = pool->type() == PoolType_AVEPOOL;
    auto builder =
        operation(pool->isGlobal() ? (average ? "reduce_mean" : "reduce_max") : (average ? "avg_pool" : "max_pool"));
    builder.input("x", input);
    if (pool->isGlobal()) {
        std::vector<int> axes;
        for (int i = 2; i < input->dimensions(); ++i) {
            axes.push_back(i);
        }
        builder.array("axes", axes).scalar("keep_dims", true);
    } else {
        int kh = std::min(pool->kernelY(), input->height());
        int kw = std::min(pool->kernelX(), input->width());
        int sh = pool->strideY(), sw = pool->strideX();
        int top = pool->padY(), left = pool->padX();
        if (pool->padType() == PoolPadType_VALID) {
            top = left = 0;
        }
        if (pool->padType() == PoolPadType_SAME) {
            top = std::max(0, (output->height() - 1) * sh + kh - input->height()) / 2;
            left = std::max(0, (output->width() - 1) * sw + kw - input->width()) / 2;
        } else if (pool->pads() && pool->pads()->size() == 4) {
            top = pool->pads()->Get(0);
            left = pool->pads()->Get(1);
        }
        // Encode the already inferred output size, including Caffe ceil mode.
        int bottom = std::max(0, (output->height() - 1) * sh + kh - input->height() - top);
        int right = std::max(0, (output->width() - 1) * sw + kw - input->width() - left);
        bool oneDimensional = input->dimensions() == 3;
        builder.array("kernel_sizes", oneDimensional ? std::vector<int>{kh} : std::vector<int>{kh, kw})
            .array("strides", oneDimensional ? std::vector<int>{sh} : std::vector<int>{sh, sw})
            .array("pad", oneDimensional ? std::vector<int>{top, bottom} : std::vector<int>{top, bottom, left, right})
            .scalar("pad_type", "custom")
            .scalar("ceil_mode", false);
        if (average) {
            builder.scalar("exclude_padding_from_average", true, "exclude_padding");
            bool include = pool->countType() == AvgPoolCountType_INCLUDE_PADDING ||
                           (pool->countType() == AvgPoolCountType_DEFAULT && pool->padType() == PoolPadType_CAFFE &&
                            !pool->pads());
            if (include && (top || bottom || left || right)) {
                std::vector<float> factors;
                for (int y = 0; y < output->height(); ++y) {
                    int iy = y * sh - top;
                    int validH = std::max(0, std::min(iy + kh, input->height()) - std::max(iy, 0));
                    int countH = std::min(iy + kh, input->height() + top) - iy;
                    for (int x = 0; x < output->width(); ++x) {
                        int ix = x * sw - left;
                        int validW = std::max(0, std::min(ix + kw, input->width()) - std::max(ix, 0));
                        int countW = std::min(ix + kw, input->width() + left) - ix;
                        factors.push_back(countH * countW > 0 ? float(validH * validW) / (countH * countW) : 0.0f);
                    }
                }
                auto pooled = builder.output(output->shape(), "exclude_pad");
                auto reshape = operation("reshape");
                reshape.array("x", factors, "counts");
                std::vector<int> shape = oneDimensional ? std::vector<int>{1, 1, output->height()}
                                                        : std::vector<int>{1, 1, output->height(), output->width()};
                reshape.shape("shape", shape, "counts_shape");
                auto factor = reshape.output(shape, "counts");
                operation("mul").input("x", pooled).input("y", factor).output(outputs);
                return NO_ERROR;
            }
        }
    }
    builder.output(outputs);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLPool, OpType_Pooling);
} // namespace MNN
