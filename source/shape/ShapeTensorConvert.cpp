//
//  ShapeTensorConvert.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {
class TensorConvertSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto& ib = inputs[0]->buffer();
        auto& ob = outputs[0]->buffer();
        if (ib.dimensions != 4 && ib.dimensions != 2) {
            return false;
        }
        MNN_ASSERT(ib.dimensions == 4);
        ob.dimensions                                         = 4;
        auto info                                             = op->main_as_TensorConvertInfo();
        auto sourceFmt                                        = info->source();
        auto destFmt                                          = info->dest();
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = destFmt;
        ob.type                                               = ib.type;

        if (sourceFmt == MNN_DATA_FORMAT_NC4HW4 && destFmt == MNN_DATA_FORMAT_NHWC) {
            ob.dim[0].extent = ib.dim[0].extent;
            ob.dim[1].extent = ib.dim[2].extent;
            ob.dim[1].flags  = 0;
            ob.dim[2].extent = ib.dim[3].extent;
            ob.dim[3].extent = ib.dim[1].extent;
        } else if (destFmt == MNN_DATA_FORMAT_NC4HW4 && sourceFmt == MNN_DATA_FORMAT_NHWC) {
            ob.dim[0].extent = ib.dim[0].extent;
            ob.dim[1].extent = ib.dim[3].extent;
            ob.dim[1].flags  = Tensor::REORDER_4;
            ob.dim[2].extent = ib.dim[1].extent;
            ob.dim[3].extent = ib.dim[2].extent;
        } else if (sourceFmt == MNN_DATA_FORMAT_NC4HW4 && destFmt == MNN_DATA_FORMAT_NCHW) {
            ob.dim[0].extent = ib.dim[0].extent;
            ob.dim[1].extent = ib.dim[1].extent;
            ob.dim[1].flags  = 0;
            ob.dim[2].extent = ib.dim[2].extent;
            ob.dim[3].extent = ib.dim[3].extent;
        } else if (sourceFmt == MNN_DATA_FORMAT_NCHW && destFmt == MNN_DATA_FORMAT_NC4HW4) {
            ob.dim[0].extent = ib.dim[0].extent;
            ob.dim[1].extent = ib.dim[1].extent;
            ob.dim[1].flags  = Tensor::REORDER_4;
            ob.dim[2].extent = ib.dim[2].extent;
            ob.dim[3].extent = ib.dim[3].extent;
        } else {
            MNN_ASSERT(false);
        }

        return true;
    }
};

REGISTER_SHAPE(TensorConvertSizeComputer, OpType_ConvertTensor);
} // namespace MNN
