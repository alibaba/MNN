//
//  GeometryGatedRMSNorm.cpp
//  MNN
//
//  Geometry for OpType_GatedRMSNorm: out = RMSNorm(x) * silu(z).
//
//  Hexagon keeps it whole when the shapes pass the same OpCommonUtils check and
//  the same channel alignment its creator uses (MNN_HEXAGON_ENABLED, compile
//  time); the creator still refuses a skel without the kernel, which then fails
//  the session rather than decomposing -- accepted because host and skel ship
//  together here.
//  Metal keeps the op whole when the native fused kernel is compiled in
//  (MNN_GATED_RMS_NORM) and OpCommonUtils::gatedRMSNormFusable accepts the
//  shapes, layout and device — keeping it whole in any other case would be
//  fatal, since the Metal creator would then reject it and no backend has a
//  fallback execution to pick it up. Every other case gets decomposed into
//  LayerNorm + SILU + MUL, which is the graph the exporter used to emit — so
//  no per-backend implementation is required for correctness. The
//  decomposition only depends on MNN_SUPPORT_TRANSFORMER_FUSE, so models
//  carrying the op still run on builds where the native kernel is disabled.
//
//  Layout note: x is [outside, inside] with the head folded into the batch axis
//  (outside = batch * heads) while z and the output are [batch, heads*inside].
//  The fused kernel absorbs that repack via its index arithmetic; the
//  decomposition reproduces it with an explicit reshape of the normalized result
//  before the multiply.
//

#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/TensorUtils.hpp"
#ifdef MNN_HEXAGON_ENABLED
#include "backend/hexagon/htp-ops-lib/include/htp_gated_rms_norm_params.h"
#endif

#define DEFAULT_ALLOCATE_SIZE 32

namespace MNN {
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
class GeometryGatedRMSNorm : public GeometryComputer {
    static std::shared_ptr<Command> _makeLnCmd(const LayerNorm* ln, Tensor* input, Tensor* output,
                                               MNN_DATA_FORMAT fmt) {
        flatbuffers::FlatBufferBuilder builder(DEFAULT_ALLOCATE_SIZE);
        std::unique_ptr<LayerNormT> lnT(ln->UnPack());
        auto lnOffset = LayerNorm::Pack(builder, lnT.get());
        OpBuilder opB(builder);
        opB.add_type(OpType_LayerNorm);
        opB.add_main(lnOffset.Union());
        opB.add_main_type(OpParameter_LayerNorm);
        opB.add_defaultDimentionFormat(fmt);
        builder.Finish(opB.Finish());
        return GeometryComputerUtils::makeCommand(builder, {input}, {output});
    }
    static std::shared_ptr<Command> _makeSiluCmd(Tensor* input, Tensor* output, MNN_DATA_FORMAT fmt) {
        flatbuffers::FlatBufferBuilder builder(DEFAULT_ALLOCATE_SIZE);
        UnaryOpBuilder unaryB(builder);
        unaryB.add_opType(UnaryOpOperation_SILU);
        auto mainOffset = unaryB.Finish().Union();
        OpBuilder opB(builder);
        opB.add_type(OpType_UnaryOp);
        opB.add_main(mainOffset);
        opB.add_main_type(OpParameter_UnaryOp);
        opB.add_defaultDimentionFormat(fmt);
        builder.Finish(opB.Finish());
        return GeometryComputerUtils::makeCommand(builder, {input}, {output});
    }
    static std::shared_ptr<Command> _makeMulCmd(Tensor* in0, Tensor* in1, Tensor* output, MNN_DATA_FORMAT fmt) {
        flatbuffers::FlatBufferBuilder builder(DEFAULT_ALLOCATE_SIZE);
        BinaryOpBuilder binaryB(builder);
        binaryB.add_opType(BinaryOpOperation_MUL);
        auto mainOffset = binaryB.Finish().Union();
        OpBuilder opB(builder);
        opB.add_type(OpType_BinaryOp);
        opB.add_main(mainOffset);
        opB.add_main_type(OpParameter_BinaryOp);
        opB.add_defaultDimentionFormat(fmt);
        builder.Finish(opB.Finish());
        return GeometryComputerUtils::makeCommand(builder, {in0, in1}, {output});
    }

public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        // Keep the op whole only when the backend's creator will actually accept it.
#ifdef MNN_GATED_RMS_NORM
        // Only OpenCL buffer mode registers a creator; image mode and
        // MNN_FORWARD_AUTO (which clears the bit) must decompose.
        const bool openCLWhole = context.forwardType() == MNN_FORWARD_OPENCL &&
                                 0 != (context.gpuMode() & MNN_GPU_MEMORY_BUFFER) &&
                                 OpCommonUtils::gatedRMSNormOpenCLOk(op, inputs, outputs);
#ifdef MNN_HEXAGON_ENABLED
        // Same shape/alignment predicate as HexagonGatedRMSNorm::supported, except externalGammaOk. The
        // geometry runs before the loader inlines gamma/beta, so they are still external here and refusing
        // them would decompose every real model -- measured on Qwen3.5-2B: GATED_RMS_NORM disappeared from
        // the profile and LAYER_NORM_PACKED went 11.6 -> 52.9 ms. The creator runs after inlining, so it
        // keeps the strict default; an op reaching it still external then fails loudly rather than
        // computing against the zero-filled gamma makeResource would hand it.
        int inside = 0;
        const bool hexagonWhole =
            context.forwardType() == MNN_FORWARD_HEXAGON &&
            OpCommonUtils::gatedRMSNormOpenCLOk(op, inputs, outputs, nullptr, &inside, nullptr, true) &&
            inside % HTP_GATED_RMS_NORM_PACK == 0;
#else
        const bool hexagonWhole = false;
#endif
        if (openCLWhole || hexagonWhole ||
            (context.forwardType() == MNN_FORWARD_METAL &&
             OpCommonUtils::gatedRMSNormFusable(op, inputs, outputs,
                                                0 != context.runtimeStatus(STATUS_SUPPORT_SIMD_GROUP_REDUCE)))) {
            std::shared_ptr<Command> cmdP(new Command);
            auto& cmd   = *cmdP;
            cmd.op      = op;
            cmd.inputs  = inputs;
            cmd.outputs = std::move(outputs);
            res.command.emplace_back(std::move(cmdP));
            return true;
        }
#endif
        auto param = op->main_as_LayerNorm();
        if (param == nullptr || inputs.size() != 2 || outputs.size() != 1) {
            return false;
        }
        auto x   = inputs[0];
        auto z   = inputs[1];
        auto out = outputs[0];
        const auto fmt = op->defaultDimentionFormat();

        // normalized = RMSNorm(x), same shape as x.
        std::shared_ptr<Tensor> normalized(
            Tensor::createDevice(x->shape(), x->getType(), x->getDimensionType()));
        // gated = silu(z), same shape as z.
        std::shared_ptr<Tensor> gated(
            Tensor::createDevice(z->shape(), z->getType(), z->getDimensionType()));
        // The normalized result viewed in z's flattened layout, so the multiply
        // is elementwise on matching shapes.
        std::shared_ptr<Tensor> normalizedFlat(
            Tensor::createDevice(z->shape(), z->getType(), z->getDimensionType()));
        if (fmt == MNN_DATA_FORMAT_NC4HW4) {
            TensorUtils::getDescribe(normalized.get())->dimensionFormat     = MNN_DATA_FORMAT_NC4HW4;
            TensorUtils::getDescribe(gated.get())->dimensionFormat          = MNN_DATA_FORMAT_NC4HW4;
            TensorUtils::getDescribe(normalizedFlat.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        }
        res.extras.emplace_back(normalized);
        res.extras.emplace_back(gated);
        res.extras.emplace_back(normalizedFlat);

        res.command.emplace_back(_makeLnCmd(param, x, normalized.get(), fmt));
        // View normalized [outside, inside] as z's flattened [1, outside*inside]
        // so the multiply is elementwise on matching shapes. The raster indexes
        // both sides logically, so the span must be the logical element count:
        // elementSize() would add NC4HW4 channel padding, and that surplus tail
        // wraps onto in-range addresses and overwrites real outputs.
        GeometryComputerUtils::makeRawAddressRef(normalizedFlat.get(), normalized.get(), 0,
                                                 (int)TensorUtils::getRawSize(normalized.get()));

        res.command.emplace_back(_makeSiluCmd(z, gated.get(), fmt));
        res.command.emplace_back(_makeMulCmd(normalizedFlat.get(), gated.get(), out, fmt));
        return true;
    }
};
static void _createGatedRMSNorm() {
    std::shared_ptr<GeometryComputer> comp(new GeometryGatedRMSNorm);
    GeometryComputer::registerGeometryComputer(comp, {OpType_GatedRMSNorm});
}
#else
static void _createGatedRMSNorm() {
}
#endif
REGISTER_GEOMETRY(GeometryGatedRMSNorm, _createGatedRMSNorm);
} // namespace MNN
