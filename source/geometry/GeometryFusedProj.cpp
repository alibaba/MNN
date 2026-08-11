//
//  GeometryFusedProj.cpp
//  MNN
//
//  Geometry for the export-time fused projection op (OpType_FusedLinear).
//  Metal, OpenCL (buffer mode), Vulkan (buffer variant) and CUDA keep the op
//  whole (native / composite executions); every other backend gets the graph
//  decomposed back into the member conv1x1 ops (+ SiLU-mul / binary RMSNorm),
//  which is exactly the unfused graph — so no per-backend implementation is
//  required for correctness.
//

#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/FusedProjCommon.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
class GeometryFusedProj : public GeometryComputer {
    static std::shared_ptr<Command> _makeCmd(std::shared_ptr<BufferStorage> storage,
                                             const std::vector<Tensor*>& inputs,
                                             const std::vector<Tensor*>& outputs) {
        std::shared_ptr<Command> cmdP(new Command);
        auto& cmd  = *cmdP;
        cmd.buffer = storage;
        cmd.op     = FusedProjCommon::opOf(storage);
        cmd.inputs = inputs;
        cmd.outputs = outputs;
        return cmdP;
    }
    static std::shared_ptr<Command> _makeConvCmd(const Convolution2D* conv, Tensor* input, Tensor* output,
                                                 MNN_DATA_FORMAT fmt) {
        return _makeCmd(FusedProjCommon::makeConvOp(conv, fmt), {input}, {output});
    }
    static std::shared_ptr<Command> _makeLnCmd(const LayerNorm* ln, Tensor* residual, Tensor* hidden,
                                               Tensor* residualOut, Tensor* normalizedOut, MNN_DATA_FORMAT fmt) {
        return _makeCmd(FusedProjCommon::makeLayerNormOp(ln, fmt), {residual, hidden}, {residualOut, normalizedOut});
    }
    static std::shared_ptr<Command> _makeMulSiluCmd(Tensor* up, Tensor* gate, Tensor* output, MNN_DATA_FORMAT fmt) {
        return _makeCmd(FusedProjCommon::makeMulSiluOp(fmt), {up, gate}, {output});
    }

    // Which backends want the op kept whole (they have a native execution for
    // it). Everything else falls through to the decomposition below. The
    // predicates come from FusedProjCommon so this gate and the creators cannot
    // disagree.
    static bool _keepWhole(const Context& context, const Op* op, size_t numInputs, size_t numOutputs) {
        if (context.forwardType() == MNN_FORWARD_METAL) {
            return FusedProjCommon::nativeEnvelopeOk(op, numInputs, numOutputs) &&
                   FusedProjCommon::allMembersAre1x1(op);
        }
        // Vulkan (buffer variant) and CUDA drive the members as child
        // executions; the runtime says whether that composite path is compiled
        // in (image-variant Vulkan and older runtimes answer 0).
        if (context.forwardType() == MNN_FORWARD_VULKAN || context.forwardType() == MNN_FORWARD_CUDA) {
            return 0 != context.runtimeStatus(STATUS_SUPPORT_FUSED_PROJ) &&
                   FusedProjCommon::compositeEnvelopeOk(op, numInputs, numOutputs);
        }
        if (context.forwardType() != MNN_FORWARD_OPENCL) {
            return false;
        }
        if (FusedProjCommon::openCLDisabled()) {
            return false;
        }
        // Only the buffer memory mode registers a creator. Keeping the op whole
        // in image mode (or under MNN_FORWARD_AUTO, which clears the bit) would
        // fail session creation outright, since no backend can take the op.
        if (0 == (context.gpuMode() & MNN_GPU_MEMORY_BUFFER)) {
            return false;
        }
        return FusedProjCommon::nativeEnvelopeOk(op, numInputs, numOutputs);
    }

public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        if (_keepWhole(context, op, inputs.size(), outputs.size())) {
            std::shared_ptr<Command> cmdP(new Command);
            auto& cmd  = *cmdP;
            cmd.op     = op;
            cmd.inputs = inputs;
            cmd.outputs = std::move(outputs);
            res.command.emplace_back(std::move(cmdP));
            return true;
        }
        auto param = op->main_as_FusedLinearParam();
        if (param == nullptr || param->convs() == nullptr) {
            return false;
        }
        const int numConvs  = (int)param->convs()->size();
        const bool isGateUp = param->act_silu_mul();
        const bool hasLn    = param->has_ln();
        const auto fmt      = op->defaultDimentionFormat();
        if (isGateUp && numConvs != 2) {
            return false;
        }
        if (!isGateUp && (numConvs < 3 || numConvs > 4)) {
            return false;
        }
        // Projection source: has_ln → normalized hidden computed by an
        // emitted binary RMSNorm (inputs [residual, hidden]); else inputs[0].
        Tensor* projInput = hasLn ? inputs[1] : inputs[0];
        const int numProjOut = isGateUp ? 1 : numConvs;
        if (hasLn) {
            if (param->ln() == nullptr || inputs.size() < 2) {
                return false;
            }
            std::shared_ptr<Tensor> normalized(Tensor::createDevice(projInput->shape(), projInput->getType(),
                                                                    projInput->getDimensionType()));
            if (fmt == MNN_DATA_FORMAT_NC4HW4) {
                TensorUtils::getDescribe(normalized.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            }
            res.extras.emplace_back(normalized);
            res.command.emplace_back(_makeLnCmd(param->ln(), inputs[0], inputs[1], outputs[numProjOut],
                                                normalized.get(), fmt));
            projInput = normalized.get();
        }
        if (!isGateUp) {
            for (int i = 0; i < numConvs; ++i) {
                res.command.emplace_back(
                    _makeConvCmd(param->convs()->GetAs<Convolution2D>(i), projInput, outputs[i], fmt));
            }
            return true;
        }
        // act_silu_mul: gate = convs[0](x), up = convs[1](x), out = up * silu(gate).
        std::shared_ptr<Tensor> gateT, upT;
        auto makeProjTensor = [&](std::shared_ptr<Tensor>& holder, int oc) {
            auto shape = projInput->shape();
            if (shape.size() >= 2) {
                shape[1] = oc;
            }
            holder.reset(Tensor::createDevice(shape, projInput->getType(), projInput->getDimensionType()));
            if (fmt == MNN_DATA_FORMAT_NC4HW4) {
                TensorUtils::getDescribe(holder.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            }
            res.extras.emplace_back(holder);
        };
        makeProjTensor(gateT, param->convs()->GetAs<Convolution2D>(0)->common()->outputCount());
        makeProjTensor(upT, param->convs()->GetAs<Convolution2D>(1)->common()->outputCount());
        res.command.emplace_back(_makeConvCmd(param->convs()->GetAs<Convolution2D>(0), projInput, gateT.get(), fmt));
        res.command.emplace_back(_makeConvCmd(param->convs()->GetAs<Convolution2D>(1), projInput, upT.get(), fmt));
        res.command.emplace_back(_makeMulSiluCmd(upT.get(), gateT.get(), outputs[0], fmt));
        return true;
    }
};
static void _createFusedProj() {
    std::shared_ptr<GeometryComputer> comp(new GeometryFusedProj);
    GeometryComputer::registerGeometryComputer(comp, {OpType_FusedLinear});
}
#else
static void _createFusedProj() {
}
#endif
REGISTER_GEOMETRY(GeometryFusedProj, _createFusedProj);
} // namespace MNN
