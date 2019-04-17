//
//  ShapeQuantizedReshape.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {
class QuantizedReshapeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto layer_param = op->main_as_QuantizedReshape();

        auto input  = inputs[0];
        auto output = outputs[0];

        const int32_t* dim_data = nullptr;
        int32_t dimSize         = 0;
        bool istflite           = (layer_param->modelFormat() == ModeFormat_TFLITE);
        if (true == istflite) {
            dimSize  = layer_param->dims()->size();
            dim_data = layer_param->dims()->data();
        } else {
            MNN_ASSERT(1 == inputs[1]->buffer().dimensions);
            auto shape      = inputs[1];
            dimSize         = shape->buffer().dim[0].extent;
            dim_data        = shape->host<int32_t>();
            auto output_min = outputs[1]->buffer();
            auto output_max = outputs[2]->buffer();

            output_min.dim[0].extent = output_min.dim[1].extent = output_min.dim[2].extent = output_min.dim[3].extent =
                1;
            output_min.dimensions    = 0;
            output_max.dim[0].extent = output_max.dim[1].extent = output_max.dim[2].extent = output_max.dim[3].extent =
                1;
            output_max.dimensions = 0;
        }

        int num_element = 1;
        for (int i = 0; i < input->buffer().dimensions; i++) {
            num_element *= input->buffer().dim[i].extent;
        }

        output->buffer().dimensions = dimSize;

        int count_non_minus1 = 1;
        for (int i = 0; i < dimSize; i++) {
            if (dim_data[i] != -1) {
                count_non_minus1 *= dim_data[i];
            }
        }

        MNN_ASSERT((num_element % count_non_minus1) == 0)

        for (int i = 0; i < dimSize; i++) {
            int shape_dim = dim_data[i];
            if (shape_dim == -1) {
                shape_dim = num_element / count_non_minus1;
            }
            output->buffer().dim[i].extent = shape_dim;
        }

        output->setType(DataType_DT_UINT8);

        return true;
    }
};

REGISTER_SHAPE(QuantizedReshapeComputer, OpType_QuantizedReshape);
} // namespace MNN
