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

class WhereSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto& ib = inputs[0]->buffer();
        auto& ob = outputs[0]->buffer();
        MNN_ASSERT(ib.type.code == halide_type_int);
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
        const int32_t* inputData = inputs[0]->host<int32_t>();
        // For compability
        if (nullptr == inputData) {
            return true;
        }
        std::vector<int32_t> trueVec;
        for (int i = 0; i < ob.dim[0].extent; i++) {
            if (inputData[i] > 0) {
                trueVec.push_back(i);
            }
        }
        if (trueVec.size() > 0) {
            ob.dim[0].extent = (int)trueVec.size();
        }
        return true;
    }
};

REGISTER_SHAPE_INPUTS(WhereSizeComputer, OpType_Where, std::vector<int>{0});
} // namespace MNN
