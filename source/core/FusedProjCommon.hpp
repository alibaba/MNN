//
//  FusedProjCommon.hpp
//  MNN
//
//  Shared member-op construction for the export-time fused projection op
//  (OpType_FusedLinear).
//
//  Two consumers build the same member ops: GeometryFusedProj, which decomposes
//  the group back into the unfused graph, and the backends that keep the op
//  whole and drive the members themselves. They must agree exactly on the op
//  conventions below, so both go through these helpers.
//
//  Conventions (do not change one side only):
//    - MUL_SILU:        out = in0 * silu(in1)  ->  in0 = up, in1 = gate
//    - binary RMSNorm:  in [residual, hidden], out [residual_out, normalized]
//

#ifndef FusedProjCommon_hpp
#define FusedProjCommon_hpp

#include <stdlib.h>
#include <memory>
#include "core/AutoStorage.h"
#include "MNN_generated.h"

namespace MNN {
namespace FusedProjCommon {

// A/B escape hatch: when set, OpenCL declines the op everywhere so the geometry
// decomposition runs instead. Both the geometry gate and the OpenCL creator
// must consult this — if they disagree, either session creation fails or
// StaticModule::preRearrangeWeights strips the member conv weights out of an op
// that the decomposition still needs them from.
inline bool openCLDisabled() {
    static const bool disabled = (nullptr != getenv("MNN_OPENCL_FUSED_PROJ_DISABLE"));
    return disabled;
}

// Serializes a builder into a BufferStorage. The returned storage owns the
// bytes; the Op pointer obtained from it stays valid for the storage lifetime.
inline std::shared_ptr<BufferStorage> finish(flatbuffers::FlatBufferBuilder& builder,
                                             flatbuffers::Offset<Op> op) {
    builder.Finish(op);
    std::shared_ptr<BufferStorage> storage(new BufferStorage);
    storage->storage = builder.ReleaseRaw(storage->allocated_size, storage->offset);
    return storage;
}

// Conv1x1 member op. externalPath is copied over when non-null, so a child
// created from this op can still resolve externally stored weights.
inline std::shared_ptr<BufferStorage> makeConvOp(const Convolution2D* conv, MNN_DATA_FORMAT fmt,
                                                 const flatbuffers::String* externalPath = nullptr) {
    flatbuffers::FlatBufferBuilder builder(1024);
    std::unique_ptr<Convolution2DT> convT(conv->UnPack());
    auto convOffset = Convolution2D::Pack(builder, convT.get());
    // Sub-objects must be finished before the parent table is started.
    flatbuffers::Offset<flatbuffers::String> pathOffset = 0;
    if (externalPath != nullptr) {
        pathOffset = builder.CreateString(externalPath->str());
    }
    OpBuilder opB(builder);
    opB.add_type(OpType_Convolution);
    opB.add_main(convOffset.Union());
    opB.add_main_type(OpParameter_Convolution2D);
    opB.add_defaultDimentionFormat(fmt);
    if (!pathOffset.IsNull()) {
        opB.add_externalPath(pathOffset);
    }
    return finish(builder, opB.Finish());
}

// MUL_SILU binary op: out = in0 * silu(in1), so in0 = up and in1 = gate.
inline std::shared_ptr<BufferStorage> makeMulSiluOp(MNN_DATA_FORMAT fmt) {
    flatbuffers::FlatBufferBuilder builder(256);
    BinaryOpBuilder binaryB(builder);
    binaryB.add_opType(BinaryOpOperation_MUL_SILU);
    auto mainOffset = binaryB.Finish().Union();
    OpBuilder opB(builder);
    opB.add_type(OpType_BinaryOp);
    opB.add_main(mainOffset);
    opB.add_main_type(OpParameter_BinaryOp);
    opB.add_defaultDimentionFormat(fmt);
    return finish(builder, opB.Finish());
}

// Binary RMSNorm: in [residual, hidden], out [residual_out, normalized].
inline std::shared_ptr<BufferStorage> makeLayerNormOp(const LayerNorm* ln, MNN_DATA_FORMAT fmt) {
    flatbuffers::FlatBufferBuilder builder(1024);
    std::unique_ptr<LayerNormT> lnT(ln->UnPack());
    auto lnOffset = LayerNorm::Pack(builder, lnT.get());
    OpBuilder opB(builder);
    opB.add_type(OpType_LayerNorm);
    opB.add_main(lnOffset.Union());
    opB.add_main_type(OpParameter_LayerNorm);
    opB.add_defaultDimentionFormat(fmt);
    return finish(builder, opB.Finish());
}

// Member/shape envelope the native executions require. The geometry keep-whole
// gate and every creator must ask the same question: an op the geometry keeps
// whole but a creator then refuses fails session creation outright, with
// StaticModule::preRearrangeWeights having already stripped the member weights.
inline bool nativeEnvelopeOk(const Op* op, size_t numInputs, size_t numOutputs) {
    auto param = op->main_as_FusedLinearParam();
    if (nullptr == param || nullptr == param->convs()) {
        return false;
    }
    const int numConvs = (int)param->convs()->size();
    if (param->act_silu_mul()) {
        if (numConvs != 2) {
            return false;
        }
    } else if (numConvs < 3 || numConvs > 4) {
        return false;
    }
    const int numProjOut = param->act_silu_mul() ? 1 : numConvs;
    if ((int)numOutputs < numProjOut) {
        return false;
    }
    if (param->has_ln() &&
        (param->ln() == nullptr || numInputs < 2 || (int)numOutputs < numProjOut + 1)) {
        return false;
    }
    return true;
}

// Metal drives each member as a MetalConvolution1x1, so a member that the
// convolution creator would build as Winograd or as the generic MetalConvolution
// must not reach it. Conditions mirror MetalConvolution1x1::isValid plus
// MetalConvolutionCreator's own early-outs; core cannot include the Metal
// header, so keep the two in step.
inline bool allMembersAre1x1(const Op* op) {
    auto param = op->main_as_FusedLinearParam();
    if (nullptr == param || nullptr == param->convs()) {
        return false;
    }
    for (int i = 0; i < (int)param->convs()->size(); ++i) {
        auto conv = param->convs()->GetAs<Convolution2D>(i);
        // MetalConvolutionCreator refuses scaleInt weights before it ever gets
        // to isValid, so keeping such a member whole would fail child creation
        // after preRearrangeWeights already stripped the weights.
        if (nullptr != conv->quanParameter() && conv->quanParameter()->has_scaleInt()) {
            return false;
        }
        auto common = conv->common();
        if (nullptr == common || common->group() > 1) {
            return false;
        }
        if (common->kernelX() != 1 || common->kernelY() != 1 || common->dilateX() != 1 ||
            common->dilateY() != 1 || common->strideX() != 1 || common->strideY() != 1 ||
            common->padX() != 0 || common->padY() != 0) {
            return false;
        }
    }
    return true;
}

inline const Op* opOf(const std::shared_ptr<BufferStorage>& storage) {
    return flatbuffers::GetRoot<Op>(storage->buffer());
}

// Every member must be an int-quantized (IDST type 1, non-scaleInt) conv: the
// only weight flavour whose child conv creation is guaranteed to succeed on
// every composite backend (Vulkan rejects scaleInt outright and its fp16-weight
// path has creator branches that may return null). Anything else stays on the
// geometry decomposition.
inline bool membersIntQuantOk(const Op* op) {
    auto param = op->main_as_FusedLinearParam();
    if (nullptr == param || nullptr == param->convs()) {
        return false;
    }
    for (int i = 0; i < (int)param->convs()->size(); ++i) {
        auto quan = param->convs()->GetAs<Convolution2D>(i)->quanParameter();
        if (nullptr == quan || quan->type() != 1 || quan->has_scaleInt()) {
            return false;
        }
    }
    return true;
}

// Keep-whole predicate for the composite (child-execution) backends: the
// geometry gate and the Vulkan / CUDA creators must all ask exactly this.
inline bool compositeEnvelopeOk(const Op* op, size_t numInputs, size_t numOutputs) {
    return nativeEnvelopeOk(op, numInputs, numOutputs) && allMembersAre1x1(op) && membersIntQuantOk(op);
}

} // namespace FusedProjCommon
} // namespace MNN

#endif /* FusedProjCommon_hpp */
