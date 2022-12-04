//
//  ShapeWhere.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
#define MNN_WHERE_OLD_VERSION

template <typename T>
int _count(Tensor* t) {
    const T* ptr = t->host<T>();
    int count = 0;
    for (int i = 0; i < t->elementSize(); i++) {
        count += (ptr[i] > 0);
    }
    return count;
}

class WhereSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto& ib = inputs[0]->buffer();
        auto& ob = outputs[0]->buffer();
        ob.dimensions = 2;
        ob.dim[0].extent = inputs[0]->elementSize();
        ob.dim[1].extent = ib.dimensions;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        outputs[0]->buffer().type = halide_type_of<int32_t>();
        auto param = op->main_as_Extra();
        if (param == nullptr) {
            // support old version
            return true;
        }
        // For zeroshape input
        if (nullptr == inputs[0]->host<void>()) {
            ob.dim[0].extent = 0;
            return true;
        }
        int count = 0;
        if (ib.type == halide_type_of<float>()) {
            count = _count<float>(inputs[0]);
        } else if (ib.type == halide_type_of<int32_t>()) {
            count = _count<int32_t>(inputs[0]);
        } else if (ib.type == halide_type_of<uint8_t>()) {
            count = _count<uint8_t>(inputs[0]);
        } else {
            return false;
        }

        if (count > 0) {
            ob.dim[0].extent = count;
        } else {
            // When no true element is found, the second demision should be kept, other than squeezed.
            ob.dimensions = 2;
            ob.dim[0].extent = 0;
            ob.dim[1].extent = ib.dimensions;
        }
        return true;
    }
};

REGISTER_SHAPE_INPUTS(WhereSizeComputer, OpType_Where, std::vector<int>{0});
} // namespace MNN
