//
//  GeometryFill.cpp
//  MNN
//
//  Created by MNN on 2020/04/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {
class GeometryFill : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        // inputs[0] is shape, inputs[1] is value
        auto input     = inputs[1];
        auto output    = outputs[0];
        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->regions.clear();
        if (0 == output->dimensions()) {
            return true;
        }
        outputDes->regions.resize(1);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;

        auto& reg   = outputDes->regions[0];
        reg.size[0] = 1;
        reg.size[1] = 1;
        for (int i = 0; i < output->dimensions(); ++i) {
            reg.size[2] *= output->length(i);
        }
        reg.src.offset    = 0;
        reg.dst.stride[2] = 1;
        reg.src.stride[2] = 0;
        reg.origin        = input;
        return true;
    }
};

class GeometryZerolike : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        // Just create empty region for raster
        auto output    = outputs[0];
        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->regions.clear();
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryFill);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Fill});
    std::shared_ptr<GeometryComputer> comp2(new GeometryZerolike);
    GeometryComputer::registerGeometryComputer(comp2, {OpType_ZerosLike, OpType_ZeroGrad});
}

REGISTER_GEOMETRY(GeometryFill, _create);

} // namespace MNN
