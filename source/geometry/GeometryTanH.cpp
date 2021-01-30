//
//  GeometryTanH.cpp
//  MNN
//
//  Created by MNN on 2020/07/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"

namespace MNN {
class GeometryTanH : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto input = inputs[0];
        auto output = outputs[0];
        auto cmd = GeometryComputerUtils::makeUnary(UnaryOpOperation_TANH, input, output);
        res.command.emplace_back(std::move(cmd));
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryTanH);
    GeometryComputer::registerGeometryComputer(comp, {OpType_TanH});
}

REGISTER_GEOMETRY(GeometryTanH, _create);

} // namespace MNN
