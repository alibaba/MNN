//
//  GeometryBinary.cpp
//  MNN
//
//  Created by MNN on 2020/05/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvertUtils.hpp"
#include "geometry/GeometryComputer.hpp"
#include "shape/SizeComputer.hpp"
namespace MNN {
class GeometryBinary : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input0     = inputs[0];
        auto input1     = inputs[1];
        auto output     = outputs[0];
        auto inputL0    = input0->elementSize();
        auto inputL1    = input1->elementSize();
        auto outputSize = output->elementSize();
        MNN_ASSERT(0 != inputL1 && 0 != inputL0 && 0 != outputSize);
        if (1 == inputL0 || 1 == inputL1) {
            // Can directly compute
            Command cmd;
            cmd.op      = op;
            cmd.inputs  = {input0, input1};
            cmd.outputs = std::move(outputs);
            res.command.emplace_back(std::move(cmd));
            return true;
        }
        // Need Broadcast or same shape
        if (outputSize != inputL0) {
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(output, newTensor.get(), true);
            newTensor->buffer().type = output->buffer().type;
            ConvertUtils::broadcastto(input0, newTensor.get());
            input0 = newTensor.get();
            res.extras.emplace_back(newTensor);
        }
        if (outputSize != inputL1) {
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(output, newTensor.get(), true);
            newTensor->buffer().type = output->buffer().type;
            ConvertUtils::broadcastto(input1, newTensor.get());
            input1 = newTensor.get();
            res.extras.emplace_back(newTensor);
        }
        Command cmd;
        cmd.op      = op;
        cmd.inputs  = {input0, input1};
        cmd.outputs = std::move(outputs);
        res.command.emplace_back(std::move(cmd));
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryBinary);
    GeometryComputer::registerGeometryComputer(comp, {OpType_BinaryOp});
}

REGISTER_GEOMETRY(GeometryBinary, _create);

} // namespace MNN
