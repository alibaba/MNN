//
//  GeometryThreshold.cpp
//  MNN
//
//  Created by MNN on 2020/07/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"

namespace MNN {
class GeometryThreshold : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto input = inputs[0];
        auto output = outputs[0];
        // compare with threshold
        std::shared_ptr<Tensor> compValue(new Tensor);
        {
            auto thresholdConst = context.allocConst(op, {}, halide_type_of<float>());
            thresholdConst->host<float>()[0] = op->main_as_ELU()->alpha();
            compValue->buffer().type = halide_type_of<int>();
            TensorUtils::copyShape(input, compValue.get(), true);
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_GREATER, input, thresholdConst.get(), compValue.get());
            res.extras.emplace_back(compValue);
            res.command.emplace_back(std::move(cmd));
        }
        // select
        {
            auto zeroConst = context.allocConst(op, {}, halide_type_of<float>());
            auto oneConst = context.allocConst(op, {}, halide_type_of<float>());
            zeroConst->host<float>()[0] = 0.0;
            oneConst->host<float>()[0] = 1.0;
            flatbuffers::FlatBufferBuilder builder;
            OpBuilder opB(builder);
            opB.add_type(OpType_Select);
            builder.Finish(opB.Finish());
            auto cmd = GeometryComputerUtils::makeCommand(builder, {compValue.get(), oneConst.get(), zeroConst.get()}, {output});
            res.command.emplace_back(std::move(cmd));
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryThreshold);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Threshold});
}

REGISTER_GEOMETRY(GeometryThreshold, _create);

} // namespace MNN
