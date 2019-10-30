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
        if (ib.dimensions <= 1) {
            return false;
        }
        auto info                                             = op->main_as_TensorConvertInfo();
        auto sourceFmt                                        = info->source();
        auto destFmt                                          = info->dest();
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = destFmt;
        ob.type                                               = ib.type;
        ob.dimensions                                         = ib.dimensions;
        
        if ((ib.dimensions == 2) ||
            (sourceFmt == MNN_DATA_FORMAT_NC4HW4 && destFmt == MNN_DATA_FORMAT_NCHW) ||
            (sourceFmt == MNN_DATA_FORMAT_NCHW && destFmt == MNN_DATA_FORMAT_NC4HW4)) {
            
            for (int i = 0; i < ib.dimensions; ++i) {
                ob.dim[i].extent = ib.dim[i].extent;
            }
            return true;
        }
        
        ob.dim[0].extent = ib.dim[0].extent;
        if (sourceFmt == MNN_DATA_FORMAT_NC4HW4 && destFmt == MNN_DATA_FORMAT_NHWC) {
            ob.dim[ob.dimensions - 1].extent = ib.dim[1].extent;
            for (int i = 1; i < ob.dimensions - 1; ++i) {
                ob.dim[i].extent = ib.dim[i + 1].extent;
            }
        } else if (destFmt == MNN_DATA_FORMAT_NC4HW4 && sourceFmt == MNN_DATA_FORMAT_NHWC) {
            ob.dim[1].extent = ib.dim[ib.dimensions - 1].extent;
            for (int i = 2; i < ob.dimensions; ++i) {
                ob.dim[i].extent = ib.dim[i - 1].extent;
            }
        } else {
            return false;
        }

        return true;
    }
};

REGISTER_SHAPE(TensorConvertSizeComputer, OpType_ConvertTensor);
} // namespace MNN
