//
//  ShapeSpaceToDepth.cpp
//  MNN
//
//  Created by MNN on 2019/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

class SpaceToDepthSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs, 
                                const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 1);
        MNN_ASSERT(outputs.size() == 1);
        if (inputs[0]->buffer().dimensions != 4) {
            MNN_ERROR("SpaceToDepth requires 4D input, got %d dimensions\n", inputs[0]->buffer().dimensions);
            return false;
        }

        const int blockSize = op->main_as_DepthSpaceParam()->blockSize();
        if (blockSize <= 0) {
            return false;
        }

        auto& ib = inputs[0]->buffer();
        auto& ob = outputs[0]->buffer();

        auto format = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        if (MNN_DATA_FORMAT_NHWC == format) {
            if (ib.dim[1].extent % blockSize != 0 || ib.dim[2].extent % blockSize != 0) {
                MNN_ERROR("SpaceToDepth: H(%d) and W(%d) must be divisible by blockSize(%d)\n",
                          ib.dim[1].extent, ib.dim[2].extent, blockSize);
                return false;
            }
        } else {
            if (ib.dim[2].extent % blockSize != 0 || ib.dim[3].extent % blockSize != 0) {
                MNN_ERROR("SpaceToDepth: H(%d) and W(%d) must be divisible by blockSize(%d)\n",
                          ib.dim[2].extent, ib.dim[3].extent, blockSize);
                return false;
            }
        }

        ob.dimensions = ib.dimensions;
        ob.type = ib.type;
        ob.dim[0].extent = ib.dim[0].extent;
        if (MNN_DATA_FORMAT_NHWC == format) {
            ob.dim[1].extent = ib.dim[1].extent / blockSize;
            ob.dim[2].extent = ib.dim[2].extent / blockSize;
            ob.dim[3].extent = ib.dim[3].extent * (blockSize * blockSize);
        } else {
            ob.dim[3].extent = ib.dim[3].extent / blockSize;
            ob.dim[2].extent = ib.dim[2].extent / blockSize;
            ob.dim[1].extent = ib.dim[1].extent * (blockSize * blockSize);
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE(SpaceToDepthSizeComputer, OpType_SpaceToDepth);

} // namespace MNN
