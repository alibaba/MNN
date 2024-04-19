//
//  GeometryDet.cpp
//  MNN
//
//  Created by MNN on 2020/04/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/Macro.h"
#include "core/OpCommonUtils.hpp"
namespace MNN {
class GeometryDet : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input = inputs[0], output = outputs[0];
        auto batch = output->elementSize(), M = input->length(input->dimensions() - 1);
        
        auto midInput = std::shared_ptr<Tensor>(Tensor::createDevice({batch, M, M}, input->getType(), input->getDimensionType()));
        auto midInDes = TensorUtils::getDescribe(midInput.get());
        midInDes->regions = {TensorUtils::makeFullSlice(input)};
        midInDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        
        auto midOutput = std::shared_ptr<Tensor>(Tensor::createDevice({batch}, output->getType(), output->getDimensionType()));
        auto outDes = TensorUtils::getDescribe(outputs[0]);
        outDes->regions = {TensorUtils::makeFullSlice(midOutput.get())};
        outDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        
        std::shared_ptr<Command> cmd(new Command);
        cmd->op = op;
        cmd->inputs.assign({midInput.get()});
        cmd->outputs.assign({midOutput.get()});
        res.command.emplace_back(std::move(cmd));
        
        res.extras.emplace_back(std::move(midInput));
        res.extras.emplace_back(std::move(midOutput));
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryDet);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Det});
}

REGISTER_GEOMETRY(GeometryDet, _create)

} // namespace MNN
