//
//  GeometryConvert.cpp
//  MNN
//
//  Created by MNN on 2020/04/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvertUtils.hpp"
#include "geometry/GeometryComputer.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
class GeometryConvert : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& buffer) const override {
        auto input  = inputs[0];
        auto output = outputs[0];
        return ConvertUtils::compute(input, output, buffer);
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryConvert);
    GeometryComputer::registerGeometryComputer(comp, {OpType_ConvertTensor});
}

REGISTER_GEOMETRY(GeometryConvert, _create);
}; // namespace MNN
