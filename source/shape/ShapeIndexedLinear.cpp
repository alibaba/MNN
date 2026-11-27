//
//  ShapeIndexedLinear.cpp
//  MNN
//
//  Created by MNN on 2026/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "shape/SizeComputer.hpp"
#include "core/IndexedLinearCommon.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
class IndexedLinearSizeComputer : public SizeComputer {
    bool onComputeSize(const Op* op, const std::vector<Tensor*>& inputs,
                       const std::vector<Tensor*>& outputs) const override {
        int experts, inputChannels, outputChannels;
        const bool routeWeightEarly = op->main_as_IndexedLinearParam() != nullptr &&
                                      op->main_as_IndexedLinearParam()->routeWeight() != 0;
        if ((int)inputs.size() != (routeWeightEarly ? 3 : 2) || outputs.size() != 1 ||
            !IndexedLinearCommon::dimensions(op, experts, inputChannels, outputChannels)) {
            return false;
        }
        auto x = inputs[0];
        auto ids = inputs[1];
        const bool routed = op->main_as_IndexedLinearParam()->inputMode() == 1;
        const int routeWeight = op->main_as_IndexedLinearParam()->routeWeight();
        if (routeWeight) {
            auto w = inputs[2];
            if (inputs.size() != 3 || !routed || ids->dimensions() != 2 ||
                ids->getType() != halide_type_of<int32_t>() || x->dimensions() != 3 ||
                x->getType() != halide_type_of<float>() || w->getType() != halide_type_of<float>() ||
                x->length(0) != ids->length(0) ||
                x->length(1) != ids->length(1) || x->length(2) != inputChannels || ids->length(1) <= 0) {
                return false;
            }
            if (routeWeight == 1) {
                if (w->dimensions() != 2 || w->length(0) != ids->length(0) || w->length(1) != ids->length(1)) {
                    return false;
                }
            } else {
                const int wDims = w->dimensions();
                const int wTokens = wDims == 2 ? w->length(0) : (wDims == 3 && w->length(0) == 1 ? w->length(1) : -1);
                const int wExperts = wDims == 2 ? w->length(1) : (wDims == 3 ? w->length(2) : -1);
                if (wTokens != ids->length(0) || wExperts != experts) {
                    return false;
                }
            }
            auto y = outputs[0];
            y->buffer().dimensions = 4;
            y->buffer().type = x->getType();
            y->setLength(0, ids->length(0));
            y->setLength(1, outputChannels);
            y->setLength(2, 1);
            y->setLength(3, 1);
            TensorUtils::getDescribe(y)->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            return true;
        }
        const int xDims = x->dimensions();
        const bool xShapeOk = routed ? (xDims == 3 && x->length(1) == ids->length(1))
                                     : (xDims == 2 || (xDims == 4 && x->length(2) == 1 && x->length(3) == 1));
        const int xChannels = xDims == 4 ? x->length(1) : (xDims > 0 ? x->length(xDims - 1) : 0);
        if (ids->dimensions() != 2 || ids->getType() != halide_type_of<int32_t>() || !xShapeOk ||
            x->getType() != halide_type_of<float>() || x->length(0) != ids->length(0) ||
            xChannels != inputChannels || ids->length(1) <= 0) {
            return false;
        }
        auto y = outputs[0];
        y->buffer().dimensions = 3;
        y->buffer().type = x->getType();
        y->setLength(0, ids->length(0));
        y->setLength(1, ids->length(1));
        y->setLength(2,
                     op->main_as_IndexedLinearParam()->fusedActivation() != -1 ? outputChannels / 2 : outputChannels);
        TensorUtils::getDescribe(y)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        return true;
    }
};
REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(IndexedLinearSizeComputer, OpType_IndexedLinear);
} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
