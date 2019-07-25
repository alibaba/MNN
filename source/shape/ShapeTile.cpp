//
//  ShapeTile.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {

class TileComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto& input    = inputs[0]->buffer();
        auto multiples = inputs[1];
        MNN_ASSERT(multiples->getType().code == halide_type_int);
        auto& output = outputs[0]->buffer();
        // Expected multiples argument to be a 1-D vector of length input.dimensions
        MNN_ASSERT(1 == multiples->buffer().dimensions)
        MNN_ASSERT(input.dimensions == multiples->buffer().dim[0].extent);
        const int inputDims = input.dimensions;
        ::memcpy(output.dim, input.dim, input.dimensions * sizeof(halide_dimension_t));
        output.dimensions = inputDims;
        output.type       = input.type;
        
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        output.dim[1].flags = 0;

        std::shared_ptr<Tensor> multipleTemp;

        // copy data from device to host if needed
        if (!multiples->host<int32_t>() && multiples->deviceId()) {
            multipleTemp.reset(Tensor::createHostTensorFromDevice(multiples, true));
            multiples = multipleTemp.get();
        }

        for (int i = 0; i < inputDims; ++i) {
            output.dim[i].extent = input.dim[i].extent * multiples->host<int32_t>()[i];
        }

        return true;
    }
};

REGISTER_SHAPE(TileComputer, OpType_Tile);

} // namespace MNN
