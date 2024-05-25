//
//  GeometryPermute.cpp
//  MNN
//
//  Created by MNN on 2020/04/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include "geometry/GeometryComputer.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
class GeometryPermute : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        int dims = inputs[0]->buffer().dimensions;
        int neworder[MNN_MAX_TENSOR_DIM];
        // get neworder
        if (op->type() == OpType_Permute) {
            auto shapeValue = op->main_as_Permute()->dims();
            if (nullptr != shapeValue) {
                for (int i = 0; i < dims; ++i) {
                    neworder[i] = shapeValue->data()[i];
                }
            } else {
                for (int i = 0; i < dims; ++i) {
                    neworder[i] = dims - i - 1;
                }
            }
        } else if (op->type() == OpType_Transpose) {
            MNN_ASSERT(inputs.size() > 1);
            auto shapeValue = inputs[1]->host<int32_t>();
            for (int i = 0; i < dims; ++i) {
                neworder[i] = shapeValue[i];
            }
        } else {
            MNN_ASSERT(false);
        }
        return GeometryComputer::ComputePermuteRegion(inputs[0], outputs[0], neworder, dims);
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryPermute);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Transpose, OpType_Permute});
}

REGISTER_GEOMETRY(GeometryPermute, _create);
}; // namespace MNN
