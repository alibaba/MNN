//
//  ShapeShape.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

class ShapeSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 <= inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto& ib = inputs[0]->buffer();
        auto& ob = outputs[0]->buffer();

        ob.dimensions = 1;
        outputs[0]->setType(DataType_DT_INT32);
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = op->defaultDimentionFormat();
        auto inputFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        if (inputFormat == MNN_DATA_FORMAT_NC4HW4 && op->defaultDimentionFormat() == MNN_DATA_FORMAT_NHWC) {
            // For compability
            ob.dim[0].extent = 4;
        } else {
            ob.dim[0].extent = ib.dimensions;
        }
        return true;
    }
};

REGISTER_SHAPE(ShapeSizeComputer, OpType_Shape);

class ShapeRasterComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 <= inputs.size());
        MNN_ASSERT(1 == outputs.size());
        outputs[0]->buffer().type = inputs[0]->buffer().type;
        auto extra  = op->main_as_Extra();
        if (!extra) {
            // copy dims
            TensorUtils::copyShape(inputs[0], outputs[0], true);
        } else {
            for (int i = 0; i < extra->attr()->size(); i++) {
                auto attr = extra->attr()->Get(i);
                if (attr->key()->str() == "shape") {
                    int len = attr->list()->i()->size();
                    outputs[0]->buffer().dimensions = len;
                    for (int j = 0; j < len; j++) {
                        outputs[0]->setLength(j, attr->list()->i()->Get(j));
                    }
                }
            }
        }
        return true;
    }
};

REGISTER_SHAPE(ShapeRasterComputer, OpType_Raster);
} // namespace MNN
