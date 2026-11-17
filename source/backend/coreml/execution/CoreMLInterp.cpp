//
//  CoreMLInterp.cpp
//  MNN
//
//  Created by MNN on 2021/05/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLInterp.hpp"
#include <cmath>
#include "backend/arm82/Arm82OptFunc.hpp"
#include "geometry/ConvertUtils.hpp"

namespace MNN {

CoreMLInterp::CoreMLInterp(MNN::Backend* b, const MNN::Op* op, const std::vector<Tensor*>& inputs,
                           const std::vector<MNN::Tensor*>& outputs)
    : CoreMLCommonExecution(b, op) {}

ErrorCode CoreMLInterp::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(!inputs.empty() && outputs.size() == 1);
    auto param = mOp->main_as_Interp();
    auto shape = inputs[0]->shape();
    auto value = mCoreMLBackend->getTensorName(inputs[0]);
    const int type = param->resizeType();
    InterpInfo info;
    info.heightScale = param->heightScale();
    info.widthScale = param->widthScale();
    info.heightOffset = param->heightOffset();
    info.widthOffset = param->widthOffset();
    if (!mCoreMLBackend->lowMemory()) {
        bool computeScale = true;
        if (inputs.size() > 1 && inputs[1]->getType().code == halide_type_float) {
            if (!inputs[1]->host<void>() || TensorUtils::getRawSize(inputs[1]) != 4) {
                return INPUT_DATA_ERROR;
            }
            std::vector<float> scales(TensorUtils::getRawSize(inputs[1]));
            auto owner = TensorUtils::getDescribeOrigin(inputs[1])->getBackend();
            if (owner && owner->type() == MNN_FORWARD_NN) {
                MNNDequantizeFP16(inputs[1]->host<int16_t>(), scales.data(), scales.size());
            } else {
                std::copy(inputs[1]->host<float>(), inputs[1]->host<float>() + scales.size(), scales.begin());
            }
            if (!std::isfinite(scales[2]) || !std::isfinite(scales[3]) || scales[2] <= 0.0f || scales[3] <= 0.0f) {
                return NOT_SUPPORT;
            }
            info.heightScale = 1.0f / scales[2];
            info.widthScale = 1.0f / scales[3];
            computeScale = false;
        }
        _ConverterInterp(param, &info, inputs[0]->width(), inputs[0]->height(), 1, outputs[0]->width(),
                         outputs[0]->height(), 1, computeScale);
    }
    if (type == 1 && shape.size() == 4 && info.heightOffset == 0.0f && info.widthOffset == 0.0f &&
        outputs[0]->height() % inputs[0]->height() == 0 && outputs[0]->width() % inputs[0]->width() == 0 &&
        info.heightScale == float(inputs[0]->height()) / outputs[0]->height() &&
        info.widthScale == float(inputs[0]->width()) / outputs[0]->width()) {
        operation("upsample_nearest_neighbor")
            .input("x", value)
            .scalar("scale_factor_height", outputs[0]->height() / inputs[0]->height(), "scale_h")
            .scalar("scale_factor_width", outputs[0]->width() / inputs[0]->width(), "scale_w")
            .output(outputs);
        return NO_ERROR;
    }
    for (int axis = 2; axis < shape.size(); ++axis) {
        int inSize = shape[axis], outSize = outputs[0]->length(axis);
        float scale = axis == 2 ? info.heightScale : info.widthScale;
        float offset = axis == 2 ? info.heightOffset : info.widthOffset;
        if (inSize == outSize && scale == 1.0f && offset == 0.0f) {
            continue;
        }
        shape[axis] = outSize;
        int taps = type == 3 ? 4 : type == 2 ? 2 : 1;
        std::string sum;
        for (int tap = 0; tap < taps; ++tap) {
            std::string tag = std::to_string(axis) + "_" + std::to_string(tap);
            std::vector<int> indices(outSize);
            std::vector<float> weights(outSize);
            for (int i = 0; i < outSize; ++i) {
                float coordinate = i * scale + offset;
                int base = int(std::floor(coordinate));
                float t = coordinate - base;
                int index = taps == 1 ? int(std::floor(coordinate + (type == 4 ? 0.499f : 0.0f)))
                                      : (type == 3 ? int(coordinate) : base) + tap - (type == 3 ? 1 : 0);
                indices[i] = std::max(0, std::min(inSize - 1, index));
                if (type == 2) {
                    weights[i] = tap == 0 ? 1.0f - t : t;
                }
                if (type == 3) {
                    float d = std::abs(t + 1.0f - tap);
                    weights[i] = d <= 1.0f ? 1.0f - 2.25f * d * d + 1.25f * d * d * d
                                           : 3.0f - 6.0f * d + 3.75f * d * d - 0.75f * d * d * d;
                }
            }
            auto sample = operation("gather")
                              .input("x", value)
                              .scalar("validate_indices", true, "validate_" + tag)
                              .array("indices", indices, "indices_" + tag)
                              .scalar("axis", axis, "axis_" + tag)
                              .output(shape, "sample_" + tag);
            if (taps > 1) {
                std::vector<int> broadcast(shape.size(), 1);
                broadcast[axis] = outSize;
                auto weight = operation("reshape")
                                  .array("x", weights, "weights_" + tag)
                                  .shape("shape", broadcast, "weight_shape_" + tag)
                                  .output(broadcast, "weight_" + tag);
                sample = operation("mul").input("x", sample).input("y", weight).output(shape, "weighted_" + tag);
            }
            if (sum.empty()) {
                sum = sample;
            } else {
                sum = operation("add").input("x", sum).input("y", sample).output(shape, "sum_" + tag);
            }
        }
        value = sum;
    }
    operation("identity").input("x", value).output(outputs);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLInterp, OpType_Interp);
} // namespace MNN
