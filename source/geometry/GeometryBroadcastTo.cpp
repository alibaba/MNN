//
//  GeometryBroadcastTo.cpp
//  MNN
//
//  Created by MNN on 2020/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvertUtils.hpp"
#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {
class GeometryBroadcastTo : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input  = inputs[0];
        auto output = outputs[0];
        bool forward = op->main() && op->main_as_Axis()->axis();
        ConvertUtils::broadcastto(input, output, forward);
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryBroadcastTo);
    GeometryComputer::registerGeometryComputer(comp, {OpType_BroadcastTo});
}

REGISTER_GEOMETRY(GeometryBroadcastTo, _create);

} // namespace MNN
