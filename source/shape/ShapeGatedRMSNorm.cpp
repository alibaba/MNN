//
//  ShapeGatedRMSNorm.cpp
//  MNN
//
//  Shape inference for OpType_GatedRMSNorm: out = RMSNorm(x) * silu(z).
//
//  The op absorbs the C4 repacks that surrounded the chain it replaces, so its
//  inputs carry different layouts: x is [outside, inside] with the head as the
//  batch axis, while z and the output are [1, outside*inside] and contiguous.
//  The output therefore follows z, not x.
//
#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
// Shape inference is registered unconditionally (under TRANSFORMER_FUSE): even
// builds without the native Metal kernel must load the op, since geometry
// decomposes it into LayerNorm + SILU + MUL as the fallback.
class GatedRMSNormSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        if (op == nullptr || inputs.size() != 2 || outputs.size() != 1) {
            return false;
        }
        auto x = inputs[0];
        auto z = inputs[1];
        auto out = outputs[0];
        if (x == nullptr || z == nullptr || out == nullptr) {
            return false;
        }
        if (x->buffer().dimensions < 2 || z->buffer().dimensions < 2) {
            return false;
        }
        auto param = op->main_as_LayerNorm();
        if (param == nullptr) {
            return false;
        }
        // Converter-folded ops keep gamma/beta in the external weight file
        // ([offset, gammaBytes, betaBytes]); exporter-era ops carry them inline.
        int gammaSize = 0;
        if (param->gamma() != nullptr) {
            gammaSize = (int)param->gamma()->size();
        } else if (param->external() != nullptr && param->external()->size() >= 2) {
            gammaSize = (int)(param->external()->data()[1] / sizeof(float));
        }
        if (gammaSize <= 0) {
            return false;
        }
        const int outside = x->length(0);
        const int inside  = x->length(1);
        const int batch   = z->length(0);
        if (outside <= 0 || inside <= 0 || batch <= 0 || (outside % batch) != 0) {
            return false;
        }
        // z / out view the same elements as x with the head folded into the
        // channel axis: x is [batch*heads, inside], z is [batch, heads*inside].
        // Decode is the batch==1 special case.
        if (z->length(1) != (outside / batch) * inside) {
            return false;
        }
        for (int i = 2; i < x->buffer().dimensions; ++i) {
            if (x->length(i) != 1) {
                return false;
            }
        }
        for (int i = 2; i < z->buffer().dimensions; ++i) {
            if (z->length(i) != 1) {
                return false;
            }
        }
        if (gammaSize != inside) {
            return false;
        }
        out->buffer().dimensions = z->buffer().dimensions;
        for (int i = 0; i < z->buffer().dimensions; ++i) {
            out->buffer().dim[i].extent = z->buffer().dim[i].extent;
        }
        out->buffer().type = z->buffer().type;
        TensorUtils::getDescribe(out)->dimensionFormat = TensorUtils::getDescribe(z)->dimensionFormat;
        return true;
    }
    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        // Two passes over x for the RMS reduction plus the gated multiply.
        return (float)inputs[0]->elementSize() * 6.f / FLOPS_M;
    }
};
REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(GatedRMSNormSizeComputer, OpType_GatedRMSNorm);
#endif
} // namespace MNN
