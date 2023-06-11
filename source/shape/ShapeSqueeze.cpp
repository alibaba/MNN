//
//  ShapeSqueeze.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
class UnSqueezeSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == outputs.size());

        const int* squeezeDim = nullptr;
        int squeezeDimSize    = 0;
        if (nullptr != op->main_as_SqueezeParam()->squeezeDims()) {
            squeezeDim     = op->main_as_SqueezeParam()->squeezeDims()->data();
            squeezeDimSize = op->main_as_SqueezeParam()->squeezeDims()->size();
        } else if (inputs.size() > 1) {
            squeezeDim     = inputs[1]->host<int>();
            squeezeDimSize = inputs[1]->elementSize();
        }
        auto& ob = outputs[0]->buffer();
        auto& ib  = inputs[0]->buffer();
        ob.dimensions = ib.dimensions + squeezeDimSize;
        uint32_t mask[MNN_MAX_TENSOR_DIM];
        ::memset(mask, 0, sizeof(mask));
        for (int i = 0; i < squeezeDimSize; i++) {
            int axis = squeezeDim[i];
            if (axis < 0) {
                axis += ob.dimensions;
            }
            mask[axis] = 1;
        }
        int oDim      = 0;
        for (int i = 0; i < ob.dimensions; i++) {
            ob.dim[i].extent = 1;
            if (mask[i] == 0) {
                ob.dim[i].extent = ib.dim[oDim].extent;
                oDim++;
            }
        }
        ob.type                                               = inputs[0]->buffer().type;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};
class SqueezeSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == outputs.size());

        const int* squeezeDim = nullptr;
        int squeezeDimSize    = 0;
        if (nullptr != op->main_as_SqueezeParam()->squeezeDims()) {
            squeezeDim     = op->main_as_SqueezeParam()->squeezeDims()->data();
            squeezeDimSize = op->main_as_SqueezeParam()->squeezeDims()->size();
        } else if (inputs.size() > 1) {
            squeezeDim     = inputs[1]->host<int>();
            squeezeDimSize = inputs[1]->elementSize();
        }
        uint32_t mask[MNN_MAX_TENSOR_DIM];
        ::memset(mask, 0, sizeof(mask));
        auto& ob = outputs[0]->buffer();
        auto& ib = inputs[0]->buffer();
        for (int i = 0; i < squeezeDimSize; i++) {
            int axis = squeezeDim[i];
            if (axis < 0) {
                axis += ib.dimensions;
            }
            if (1 != ib.dim[axis].extent) {
                MNN_ERROR("Cannot Squeeze dim[%d], 1 is expected, %d is got. input shape:", axis, ib.dim[axis].extent);
                inputs[0]->printShape();
                return false;
            }
            mask[axis] = 1;
        }
        if (squeezeDimSize == 0) {
            for (int i = 0; i < ib.dimensions; ++i) {
                if (ib.dim[i].extent == 1) {
                    mask[i] = 1;
                    ++squeezeDimSize;
                }
            }
        }
        // in = Tensor(shape=())
        // out = Squeeze(in) should also returns a tensor with shape=(), but
        // the `squeezeDimSize` and `ib.dimensions` are all 0.
        MNN_ASSERT(squeezeDimSize <= ib.dimensions);

        ob.dimensions = ib.dimensions - squeezeDimSize;
        int oDim      = 0;
        for (int i = 0; i < ib.dimensions; i++) {
            if (mask[i] == 0) {
                ob.dim[oDim].extent = ib.dim[i].extent;
                oDim++;
            }
        }
        ob.type                                               = inputs[0]->buffer().type;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
};

REGISTER_SHAPE(SqueezeSizeComputer, OpType_Squeeze);
REGISTER_SHAPE(UnSqueezeSizeComputer, OpType_Unsqueeze);
} // namespace MNN
