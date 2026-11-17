//
//  CoreMLLayerNorm.cpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLLayerNorm.hpp"

namespace MNN {

CoreMLLayerNorm::CoreMLLayerNorm(MNN::Backend* b, const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<MNN::Tensor*>& outputs)
    : CoreMLCommonExecution(b, op) {
    auto layerNormParam = mOp->main_as_LayerNorm();
    auto axis = layerNormParam->axis();
    mRMS = layerNormParam->useRMSNorm();
    mEpsilon = layerNormParam->epsilon();
    if (layerNormParam->gamma() && layerNormParam->beta() &&
        layerNormParam->gamma()->size() != layerNormParam->beta()->size()) {
        mValid = false;
        return;
    }
    for (int i = 0; axis && i < axis->size(); ++i) {
        mAxes.push_back(axis->data()[i]);
    }
    if (mAxes.empty()) {
        mAxes.push_back(-1);
    }
    if (!mRMS) {
        return;
    }
    mGamma = mCoreMLBackend->existingWeight(mName + "_gamma");
    mBeta = mCoreMLBackend->existingWeight(mName + "_beta");
    if (mGamma.empty() && layerNormParam->gamma()) {
        auto gamma = layerNormParam->gamma()->data();
        int size = layerNormParam->gamma()->size();
        mGamma = buildBlob(gamma, size * sizeof(float), CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32, {size},
                           "gamma");
        if (layerNormParam->beta()) {
            mBeta = buildBlob(layerNormParam->beta()->data(), size * sizeof(float),
                              CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32, {size}, "beta");
        }
    }
}

ErrorCode CoreMLLayerNorm::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    auto x = mCoreMLBackend->getTensorName(inputs[0]);
    auto reductionAxes = mAxes;
    bool packed = TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
    if (packed) {
        reductionAxes = {1};
    }
    if (mRMS) {
        auto shape = inputs[0]->shape();
        const auto originalShape = shape;
        auto param = mOp->main_as_LayerNorm();
        if (!packed && param->group() > 1) {
            int outer = shape[0] * param->group();
            int count = TensorUtils::getRawSize(inputs[0]);
            if (outer <= 0 || count % outer) {
                return NOT_SUPPORT;
            }
            shape = {outer, count / outer};
            reductionAxes = {1};
            x = operation("reshape").input("x", x).shape("shape", shape, "group_shape").output(shape, "groups");
        }
        auto reduced = shape;
        std::vector<int> broadcast(shape.size(), 1);
        for (auto axis : reductionAxes) {
            if (axis < 0) {
                axis += shape.size();
            }
            if (axis < 0 || axis >= shape.size()) {
                return NOT_SUPPORT;
            }
            reduced[axis] = 1;
            broadcast[axis] = shape[axis];
        }
        auto axes = buildArray(reductionAxes, "axes");
        // Accumulating squares in fp32 avoids fp16 underflow in Q/K norms.
        const auto fp32 = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32;
        auto input =
            operation("cast").input("x", x).scalar("dtype", "fp32", "accum_dtype").output(shape, "accum_input", fp32);
        auto variance = operation("square").input("x", input).output(shape, "square", fp32);
        variance = operation("reduce_mean")
                       .input("x", variance)
                       .input("axes", axes)
                       .scalar("keep_dims", true)
                       .output(reduced, "variance", fp32);
        variance = operation("add")
                       .input("x", variance)
                       .scalarFloat32("y", mEpsilon, "accum_epsilon")
                       .output(reduced, "variance_epsilon", fp32);
        auto inverse = operation("sqrt").input("x", variance).output(reduced, "inverse", fp32);
        auto normalized = operation("real_div").input("x", input).input("y", inverse).output(shape, "normalized", fp32);
        normalized = operation("cast")
                         .input("x", normalized)
                         .scalar("dtype", "fp16", "output_dtype")
                         .output(shape, "normalized_half");
        if (!mGamma.empty()) {
            auto gamma = operation("reshape")
                             .input("x", mGamma)
                             .shape("shape", broadcast, "gamma_shape")
                             .output(broadcast, "gamma_broadcast");
            normalized = operation("mul").input("x", normalized).input("y", gamma).output(shape, "scaled");
        }
        if (!mBeta.empty()) {
            auto beta = operation("reshape")
                            .input("x", mBeta)
                            .shape("shape", broadcast, "beta_shape")
                            .output(broadcast, "beta_broadcast");
            normalized = operation("add").input("x", normalized).input("y", beta).output(shape, "biased");
        }
        operation("reshape").input("x", normalized).shape("shape", originalShape, "output_shape").output(outputs);
    } else {
        auto param = mOp->main_as_LayerNorm();
        auto shape = inputs[0]->shape();
        int inner = 1, outer = 1;
        if (packed) {
            inner = shape[1];
            std::vector<int> perm = {0};
            for (int i = 2; i < shape.size(); ++i) {
                perm.push_back(i);
            }
            perm.push_back(1);
            std::vector<int> transposed;
            for (int i : perm) {
                transposed.push_back(shape[i]);
            }
            x = operation("transpose")
                    .input("x", x)
                    .array("perm", perm, "norm_perm")
                    .output(transposed, "channel_last");
        } else if (param->group() > 1) {
            inner = TensorUtils::getRawSize(inputs[0]) / (shape[0] * param->group());
        } else {
            // Geometry moves the reduction dimensions to a contiguous suffix.
            for (int i = shape.size() - mAxes.size(); i < shape.size(); ++i) {
                inner *= shape[i];
            }
        }
        outer = TensorUtils::getRawSize(inputs[0]) / inner;
        std::vector<int> matrix = {outer, inner};
        x = operation("reshape").input("x", x).shape("shape", matrix, "matrix_shape").output(matrix, "matrix");
        auto builder = operation("layer_norm");
        builder.input("x", x).array<int>("axes", {-1}, "last_axis").scalar("epsilon", mEpsilon);
        if (param->gamma()) {
            builder.array("gamma", std::vector<float>(param->gamma()->begin(), param->gamma()->end()));
        }
        if (param->beta()) {
            builder.array("beta", std::vector<float>(param->beta()->begin(), param->beta()->end()));
        }
        auto result = builder.output(matrix, "normalized");
        auto restore = operation("reshape");
        restore.input("x", result);
        if (!packed) {
            restore.shape("shape", outputs[0]->shape(), "output_shape").output(outputs);
        } else {
            std::vector<int> transposed = {shape[0]};
            for (int i = 2; i < shape.size(); ++i) {
                transposed.push_back(shape[i]);
            }
            transposed.push_back(shape[1]);
            restore.shape("shape", transposed, "restored_shape");
            result = restore.output(transposed, "restored");
            std::vector<int> perm = {0, int(shape.size()) - 1};
            for (int i = 1; i + 1 < shape.size(); ++i) {
                perm.push_back(i);
            }
            operation("transpose").input("x", result).array("perm", perm, "output_perm").output(outputs);
        }
    }
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLLayerNorm, OpType_LayerNorm);
} // namespace MNN
