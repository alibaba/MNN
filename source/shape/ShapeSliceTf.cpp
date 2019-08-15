//
//  ShapeSliceTf.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {

class SliceTfComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 3);
        MNN_ASSERT(outputs.size() == 1);

        auto input = inputs[0];
        // these two inputs should be const
        auto begin_tensor = inputs[1];
        auto size_tensor  = inputs[2];

        std::shared_ptr<Tensor> realBeginTensor;
        std::shared_ptr<Tensor> sizeTensor;

        // copy data from device to host if needed
        if (!begin_tensor->host<int32_t>() && begin_tensor->deviceId()) {
            realBeginTensor.reset(Tensor::createHostTensorFromDevice(begin_tensor, true));
            begin_tensor = realBeginTensor.get();
        }
        if (!size_tensor->host<int32_t>() && size_tensor->deviceId()) {
            sizeTensor.reset(Tensor::createHostTensorFromDevice(size_tensor, true));
            size_tensor = sizeTensor.get();
        }

        MNN_ASSERT(begin_tensor->buffer().dimensions == 1);
        MNN_ASSERT(size_tensor->buffer().dimensions == 1);
        MNN_ASSERT(input->buffer().dimensions >= 1);
        MNN_ASSERT(input->buffer().dimensions == begin_tensor->buffer().dim[0].extent);
        MNN_ASSERT(input->buffer().dimensions == size_tensor->buffer().dim[0].extent);

        auto output                 = outputs[0];
        output->buffer().dimensions = input->buffer().dimensions;
        output->buffer().type       = input->buffer().type;
        int dim                     = 0;
        for (int i = 0; i < input->buffer().dimensions; i++) {
            dim = size_tensor->host<int32_t>()[i];
            if (dim == -1 ) {
                dim = input->buffer().dim[i].extent - begin_tensor->host<int32_t>()[i];
            }
            // size <= 0, this ouput is not useful, set the dimendsions 0
            if (dim <= 0) {
                output->buffer().dimensions = 0;
                break;
            }
            output->buffer().dim[i].extent = dim;
        }

        return true;
    }
};

REGISTER_SHAPE(SliceTfComputer, OpType_SliceTf);
} // namespace MNN
