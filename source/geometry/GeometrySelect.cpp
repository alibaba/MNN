//
//  GeometrySelect.cpp
//  MNN
//
//  Created by MNN on 2020/05/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvertUtils.hpp"
#include "geometry/GeometryComputer.hpp"
#include "shape/SizeComputer.hpp"
namespace MNN {
class GeometrySelect : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input0     = inputs[0];
        auto input1     = inputs[1];
        auto input2     = inputs[2];
        auto output     = outputs[0];
        auto inputL0    = input0->elementSize();
        auto inputL1    = input1->elementSize();
        auto inputL2    = input1->elementSize();
        auto outputSize = output->elementSize();
        // Need Broadcast or same shape
        if (outputSize != inputL0) {
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(output, newTensor.get(), true);
            newTensor->buffer().type = input0->buffer().type;
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
        if (outputSize != inputL2) {
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(output, newTensor.get(), true);
            newTensor->buffer().type = output->buffer().type;
            ConvertUtils::broadcastto(input2, newTensor.get());
            input2 = newTensor.get();
            res.extras.emplace_back(newTensor);
        }
        std::shared_ptr<Command> cmdP(new Command);
        auto& cmd = *cmdP;
        cmd.op      = op;
        cmd.inputs  = {input0, input1, input2};
        cmd.outputs = std::move(outputs);
        res.command.emplace_back(std::move(cmdP));
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometrySelect);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Select});
}

REGISTER_GEOMETRY(GeometrySelect, _create);

} // namespace MNN
