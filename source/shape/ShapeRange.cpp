//
//  ShapeRange.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "core/SizeComputer.hpp"
#include "math.h"

namespace MNN {

template <typename T>
static int computeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    Tensor* start_in = inputs[0];
    Tensor* limit_in = inputs[1];
    Tensor* delta_in = inputs[2];

    MNN_ASSERT((1 == start_in->buffer().dimensions) || (0 == start_in->buffer().dimensions));
    MNN_ASSERT((1 == limit_in->buffer().dimensions) || (0 == limit_in->buffer().dimensions));
    MNN_ASSERT((1 == delta_in->buffer().dimensions) || (0 == delta_in->buffer().dimensions));

    const float start = (float)start_in->host<T>()[0];
    const float limit = (float)limit_in->host<T>()[0];
    const float delta = (float)delta_in->host<T>()[0];

    MNN_ASSERT(0 != delta);
    if (delta > 0) {
        MNN_ASSERT(limit >= start);
    } else {
        MNN_ASSERT(start >= limit);
    }

    int32_t size = ceilf(fabsf((limit - start) / delta));
    return (int)size;
}

class RangeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 3);
        auto type       = op->main_as_Range()->Tidx();
        int output_size = 0;
        switch (type) {
            case DataType_DT_INT32:
            case DataType_DT_INT64:
                output_size = computeSize<int32_t>(op, inputs, outputs);
                outputs[0]->setType(MNN::DataType_DT_INT32);
                break;
            case DataType_DT_FLOAT:
            case DataType_DT_DOUBLE:
                output_size = computeSize<float>(op, inputs, outputs);
                outputs[0]->setType(MNN::DataType_DT_FLOAT);
                break;
            default:
                MNN_ASSERT(false); // unsupported type
        }
        outputs[0]->buffer().dimensions    = 1;
        outputs[0]->buffer().dim[0].extent = output_size;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NHWC;

        return true;
    }
};

REGISTER_SHAPE_INPUTS(RangeComputer, OpType_Range, (std::vector<int>{0, 1, 2}));
} // namespace MNN
