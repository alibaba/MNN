//
//  ShapeFusedProj.cpp
//  MNN
//
//  Shape inference for the export-time fused projection ops
//  (OpType_FusedLinear).
//
#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
class FusedProjSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        if (op == nullptr || inputs.empty() || outputs.empty() || inputs[0] == nullptr) {
            return false;
        }
        auto param = op->main_as_FusedLinearParam();
        if (param == nullptr || param->convs() == nullptr || param->convs()->size() == 0) {
            return false;
        }
        const int numConvs  = (int)param->convs()->size();
        const bool isGateUp = param->act_silu_mul();
        const bool hasLn    = param->has_ln();
        if (param->row_mul()) {
            if (numConvs != 1 || isGateUp || hasLn || inputs.size() != 2 || outputs.size() != 1 ||
                inputs[1] == nullptr || inputs[0]->dimensions() != 4 || inputs[1]->dimensions() != 4 ||
                inputs[0]->getType() != halide_type_of<float>() || inputs[1]->getType() != inputs[0]->getType() ||
                TensorUtils::getDescribe(inputs[0])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
                TensorUtils::getDescribe(inputs[1])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
                inputs[0]->length(0) != inputs[1]->length(0) || inputs[1]->length(1) != 1 ||
                inputs[0]->length(2) != 1 || inputs[0]->length(3) != 1 ||
                inputs[1]->length(2) != 1 || inputs[1]->length(3) != 1) {
                return false;
            }
            auto conv = param->convs()->GetAs<Convolution2D>(0);
            if (conv == nullptr || conv->common() == nullptr || conv->common()->outputCount() <= 0) {
                return false;
            }
            auto common = conv->common();
            if (common->group() != 1 || common->kernelX() != 1 || common->kernelY() != 1 ||
                common->strideX() != 1 || common->strideY() != 1 || common->dilateX() != 1 ||
                common->dilateY() != 1 || common->padX() != 0 || common->padY() != 0 ||
                common->inputCount() != inputs[0]->length(1) || inputs[0]->length(0) <= 0) {
                return false;
            }
            TensorUtils::copyShape(inputs[0], outputs[0], true);
            outputs[0]->setLength(1, conv->common()->outputCount());
            outputs[0]->buffer().type = inputs[0]->getType();
            return true;
        }
        if (isGateUp && numConvs != 2) {
            return false;
        }
        // Non-silu multi-output: qkv fusion has 3-4 members; a 2-member op is a
        // gate/up projection whose SiLU-mul tail was materialized as plain ops
        // (split-LoRA patcher does this to inject LoRA between GEMM and SiLU).
        if (!isGateUp && (numConvs < 2 || numConvs > 4)) {
            return false;
        }
        if (hasLn && (inputs.size() < 2 || inputs[1] == nullptr)) {
            return false;
        }
        const int numProjOut = isGateUp ? 1 : numConvs;
        if ((int)outputs.size() != numProjOut + (hasLn ? 1 : 0)) {
            return false;
        }
        // has_ln: inputs = [residual, hidden] (binary RMSNorm convention);
        // else inputs = [x]. x is the projection source.
        auto x = hasLn ? inputs[1] : inputs[0];
        // Projections are conv1x1: same spatial shape as x, channel = outputCount.
        for (int i = 0; i < numProjOut; ++i) {
            auto out = outputs[i];
            if (out == nullptr) {
                return false;
            }
            auto conv = param->convs()->GetAs<Convolution2D>(i);
            if (conv == nullptr || conv->common() == nullptr || conv->common()->outputCount() <= 0) {
                return false;
            }
            out->buffer().dimensions = x->buffer().dimensions;
            for (int d = 0; d < x->buffer().dimensions; ++d) {
                out->buffer().dim[d].extent = x->buffer().dim[d].extent;
            }
            if (out->buffer().dimensions >= 2) {
                out->buffer().dim[1].extent = conv->common()->outputCount();
            }
            out->buffer().type = x->buffer().type;
            TensorUtils::getDescribe(out)->dimensionFormat = TensorUtils::getDescribe(x)->dimensionFormat;
        }
        if (hasLn) {
            auto resOut = outputs[numProjOut];
            auto resIn  = inputs[0];
            if (resOut == nullptr) {
                return false;
            }
            resOut->buffer().dimensions = resIn->buffer().dimensions;
            for (int d = 0; d < resIn->buffer().dimensions; ++d) {
                resOut->buffer().dim[d].extent = resIn->buffer().dim[d].extent;
            }
            resOut->buffer().type = resIn->buffer().type;
            TensorUtils::getDescribe(resOut)->dimensionFormat = TensorUtils::getDescribe(resIn)->dimensionFormat;
        }
        return true;
    }
    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto param = op->main_as_FusedLinearParam();
        if (param == nullptr || param->convs() == nullptr || inputs.empty()) {
            return 0.f;
        }
        auto x = inputs[0];
        float tokens = 1.f;
        for (int d = 0; d < x->buffer().dimensions; ++d) {
            if (d == 1) {
                continue; // channel
            }
            tokens *= (float)x->buffer().dim[d].extent;
        }
        float flops = 0.f;
        for (int i = 0; i < (int)param->convs()->size(); ++i) {
            auto conv = param->convs()->GetAs<Convolution2D>(i);
            if (conv == nullptr || conv->common() == nullptr) {
                continue;
            }
            flops += 2.f * tokens * (float)conv->common()->inputCount() * (float)conv->common()->outputCount();
        }
        if (param->act_silu_mul() && param->convs()->size() >= 1) {
            flops += 4.f * tokens * (float)param->convs()->GetAs<Convolution2D>(0)->common()->outputCount();
        }
        if (param->has_ln()) {
            flops += 6.f * (float)x->elementSize();
        }
        if (param->row_mul()) {
            flops += tokens * (float)param->convs()->GetAs<Convolution2D>(0)->common()->outputCount();
        }
        return flops / FLOPS_M;
    }
};
REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(FusedProjSizeComputer, OpType_FusedLinear);
#endif
} // namespace MNN
