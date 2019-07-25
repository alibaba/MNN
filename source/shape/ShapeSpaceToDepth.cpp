//
//  ShapeSpaceToDepth.cpp
//  MNN
//
//  Created by MNN on 2019/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {

class SpaceToDepthSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs, 
                                const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 1);
        MNN_ASSERT(outputs.size() == 1);
        MNN_ASSERT(inputs[0]->buffer().dimensions == 4);

        // here only implement NHWC
        // TODO: implement NC4HW4
        const int blockSize = op->main_as_DepthSpaceParam()->blockSize();
        MNN_ASSERT(blockSize > 1);
        MNN_ASSERT(inputs[0]->buffer().dim[1].extent % blockSize == 0);
        MNN_ASSERT(inputs[0]->buffer().dim[2].extent % blockSize == 0);

        auto& ib = inputs[0]->buffer();
        auto& ob = outputs[0]->buffer();

        ob.dimensions = ib.dimensions;
        ob.dim[0].extent = ib.dim[0].extent;
        ob.dim[1].extent = ib.dim[1].extent / blockSize;
        ob.dim[2].extent = ib.dim[2].extent / blockSize;
        ob.dim[3].extent = ib.dim[3].extent * (blockSize * blockSize);

        return true;
    }
};

REGISTER_SHAPE(SpaceToDepthSizeComputer, OpType_SpaceToDepth);

} // namespace MNN
