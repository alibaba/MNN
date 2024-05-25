//
//  GeometryTopK.cpp
//  MNN
//
//  Created by MNN on 2020/06/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <numeric>
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {
class GeometryTopK : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        if (outputs.size() != 2 || inputs.size() < 2 || inputs.size() > 3) {
            MNN_ERROR("TopK should have 2 output and 2~3 input, get %lu in and %lu out\n", inputs.size(), outputs.size());
            return false;
        }
        int numAxes = inputs[0]->dimensions(), axis = numAxes - 1;
        if (inputs.size() == 3) {
            axis = inputs[2]->host<int32_t>()[0];
            if (axis < 0) {
                axis += numAxes;
            }
        }
        if (axis == numAxes - 1) {
            std::shared_ptr<Command> cmdP(new Command);
            auto& cmd = *cmdP;
            cmd.op      = op;
            cmd.inputs.assign({inputs[0], inputs[1]});
            cmd.outputs = std::move(outputs);
            res.command.emplace_back(std::move(cmdP));
            return true;
        }
        if (inputs[1]->host<int32_t>() == nullptr || inputs[2]->host<int32_t>() == nullptr) {
            MNN_ERROR("Invalid k or axis\n");
            return false;
        }
        int k = inputs[1]->host<int32_t>()[0];
        auto shape = inputs[0]->shape();
        int outside = std::accumulate(shape.begin(), shape.begin() + axis, 1, [](int a, int b) { return a * b; });
        int inside = std::accumulate(shape.begin() + axis + 1, shape.end(), 1, [](int a, int b) { return a * b; });
        std::shared_ptr<Tensor> transInput, transVal, transInd;
        { // transpose TopK's axis to last axis
            transInput.reset(Tensor::createDevice({outside * inside, shape[axis]}, inputs[0]->getType(), inputs[0]->getDimensionType()));
            auto des        = TensorUtils::getDescribe(transInput.get());
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            Tensor::InsideDescribe::Region reg;
            reg.origin = inputs[0];
            reg.src.stride[0] = reg.dst.stride[0] = inside * shape[axis];
            reg.src.stride[2] = inside;
            reg.dst.stride[1] = shape[axis];
            reg.size[0] = outside;
            reg.size[1] = inside;
            reg.size[2] = shape[axis];
            des->regions.assign({reg});
            res.extras.emplace_back(transInput);
        }
        { // transpose TopK's axis from last axis
            transVal.reset(Tensor::createDevice({outside * inside, k}, outputs[0]->getType(), outputs[0]->getDimensionType()));
            transInd.reset(Tensor::createDevice({outside * inside, k}, outputs[1]->getType(), outputs[1]->getDimensionType()));
            Tensor::InsideDescribe::Region reg;
            reg.src.stride[0] = reg.dst.stride[0] = inside * k;
            reg.src.stride[2] = k;
            reg.dst.stride[1] = inside;
            reg.size[0] = outside;
            reg.size[1] = k;
            reg.size[2] = inside;
            auto des = TensorUtils::getDescribe(outputs[0]);
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            reg.origin = transVal.get();
            des->regions.assign({reg});
            res.extras.emplace_back(transVal);
            des = TensorUtils::getDescribe(outputs[1]);
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            reg.origin = transInd.get();
            des->regions.assign({reg});
            res.extras.emplace_back(transInd);
        }
        { // do TopK on last axis
            std::shared_ptr<Command> cmdP(new Command);
            auto& cmd = *cmdP;
            cmd.op      = op;
            cmd.inputs.assign({transInput.get(), inputs[1]});
            cmd.outputs.assign({transVal.get(), transInd.get()});
            res.command.emplace_back(std::move(cmdP));
        }
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryTopK);
    GeometryComputer::registerGeometryComputer(comp, {OpType_TopKV2});
}

REGISTER_GEOMETRY(GeometryTopK, _create);

} // namespace MNN
