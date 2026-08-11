//
//  GeometryGatedRMSNorm.cpp
//  MNN
//
//  Geometry for OpType_GatedRMSNorm: out = RMSNorm(x) * silu(z).
//
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
//  Layout note: x is [outside, inside] (head as batch) while z and the output
//  are [1, outside*inside]. The fused kernel absorbs that repack via its index
//  arithmetic; the decomposition reproduces it with an explicit reshape of the
//  normalized result before the multiply.
//

#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/TensorUtils.hpp"

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
        // Keep the op whole only when the Metal creator will actually accept it:
        // both sides ask the same predicate, so they cannot drift apart. It
        // covers batch (decode only), channel alignment, layout and the device's
        // simdgroup-reduce support; anything else falls through to the
        // decomposition below, like every other backend.
#ifdef MNN_GATED_RMS_NORM
        if (context.forwardType() == MNN_FORWARD_METAL &&
            OpCommonUtils::gatedRMSNormFusable(op, inputs, outputs,
                                               0 != context.runtimeStatus(STATUS_SUPPORT_SIMD_GROUP_REDUCE))) {
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
        // so the multiply is elementwise on matching shapes.
        GeometryComputerUtils::makeRawAddressRef(normalizedFlat.get(), normalized.get(), 0,
                                                normalized->elementSize());

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
