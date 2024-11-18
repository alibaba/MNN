//
//  GeometryConvert.cpp
//  MNN
//
//  Created by MNN on 2020/04/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvertUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/TensorUtils.hpp"
#include "GeometryComputer.hpp"
#include "GeometryComputerUtils.hpp"
namespace MNN {
class GeometryConvert : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& buffer) const override {
        if (op->type() == OpType_ConvertTensor) {
            auto input  = inputs[0];
            auto output = outputs[0];
            return ConvertUtils::compute(input, output, buffer);
        }
        MNN_ASSERT(op->type() == OpType_CastLike);
        auto input  = inputs[0];
        auto type  = OpCommonUtils::convertDataType(inputs[1]->getType());
        auto output = outputs[0];
        flatbuffers::FlatBufferBuilder builder;
        CastParamBuilder builder_(builder);
        builder_.add_dstT(type);
        auto mainOffset = builder_.Finish().Union();
        OpBuilder opB(builder);
        opB.add_type(OpType_Cast);
        opB.add_main(mainOffset);
        opB.add_main_type(OpParameter_CastParam);
        builder.Finish(opB.Finish());
        auto cmd = GeometryComputerUtils::makeCommand(builder, {input}, outputs);
        buffer.command.emplace_back(cmd);
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryConvert);
    GeometryComputer::registerGeometryComputer(comp, {OpType_ConvertTensor, OpType_CastLike});
}

REGISTER_GEOMETRY(GeometryConvert, _create);
}; // namespace MNN
