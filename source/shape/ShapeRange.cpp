//
//  ShapeRange.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
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

    const T start = start_in->host<T>()[0];
    const T limit = limit_in->host<T>()[0];
    const T delta = delta_in->host<T>()[0];

    MNN_ASSERT(0 != delta);
    if (delta > 0) {
        MNN_ASSERT(limit >= start);
    } else {
        MNN_ASSERT(start >= limit);
    }

    int64_t size = (std::is_integral<T>::value ? ((abs(limit - start) + abs(delta) - 1) / abs(delta))
                                               : ceil(abs((limit - start) / delta)));
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
        return true;
    }
};

REGISTER_SHAPE(RangeComputer, OpType_Range);
} // namespace MNN
