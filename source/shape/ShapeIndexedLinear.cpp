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
        if (inputs.size() != 2 || outputs.size() != 1 ||
            !IndexedLinearCommon::dimensions(op, experts, inputChannels, outputChannels)) {
            return false;
        }
        auto x = inputs[0];
        auto ids = inputs[1];
        const bool routed = op->main_as_IndexedLinearParam()->inputMode() == 1;
        if (ids->dimensions() != 2 || ids->getType() != halide_type_of<int32_t>() ||
            x->dimensions() != (routed ? 3 : 2) || x->getType() != halide_type_of<float>() ||
            x->length(0) != ids->length(0) || x->length(x->dimensions() - 1) != inputChannels || ids->length(1) <= 0 ||
            (routed && x->length(1) != ids->length(1))) {
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
