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
        if (isGateUp && numConvs != 2) {
            return false;
        }
        if (!isGateUp && (numConvs < 3 || numConvs > 4)) {
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
        return flops / FLOPS_M;
    }
};
REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(FusedProjSizeComputer, OpType_FusedLinear);
#endif
} // namespace MNN
