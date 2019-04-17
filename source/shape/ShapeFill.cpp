//
//  ShapeFill.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {

class FillComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto input0 = inputs[0], output0 = outputs[0];
        MNN_ASSERT(inputs.size() == 2);
        MNN_ASSERT(input0->buffer().dimensions == 1);
        std::shared_ptr<Tensor> tempInput0;

        // copy data from device to host if needed
        if (!input0->host<int32_t>() && input0->deviceId()) {
            tempInput0.reset(Tensor::createHostTensorFromDevice(input0, true));
            input0 = tempInput0.get();
        }

        output0->buffer().dimensions = input0->buffer().dim[0].extent;
        // TODO
        output0->setType(MNN::DataType_DT_INT32);
        for (int i = 0; i < input0->buffer().dim[0].extent; i++) {
            output0->buffer().dim[i].extent = input0->host<int32_t>()[i];
        }

        return true;
    }
};

REGISTER_SHAPE(FillComputer, OpType_Fill);
} // namespace MNN
