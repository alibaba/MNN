//
//  ShapeSlice.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {

class SliceComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(2 <= outputs.size());
        auto slice = op->main_as_Slice();

        auto& input = inputs[0]->buffer();

        int axis = slice->axis();
        if (axis < 0) {
            axis += input.dimensions;
        }

        if (MNN::NetSource_CAFFE == slice->sourceType()) {
            // caffe Slice
            int previous = 0;
            for (int i = 0; i < slice->slicePoints()->size(); ++i) {
                int sliceIndex    = slice->slicePoints()->data()[i];
                auto& output      = outputs[i]->buffer();
                output.dimensions = input.dimensions;
                ::memcpy(output.dim, input.dim, input.dimensions * sizeof(halide_dimension_t));
                output.type             = input.type;
                output.dim[axis].extent = sliceIndex - previous;
                previous                = sliceIndex;
            }

            // Compute Last
            auto& output = outputs[outputs.size() - 1]->buffer();
            ::memcpy(output.dim, input.dim, input.dimensions * sizeof(halide_dimension_t));

            output.dim[axis].extent = input.dim[axis].extent - previous;
        } else {
            // tensorflow Split
            if (1 == slice->slicePoints()->size()) {
                // scalar
                const int numSplits = slice->slicePoints()->data()[0];
                MNN_ASSERT(numSplits == outputs.size());
                MNN_ASSERT(0 == input.dim[axis].extent % numSplits);
                const int splitDim = input.dim[axis].extent / numSplits;
                for (int i = 0; i < numSplits; i++) {
                    auto& output      = outputs[i]->buffer();
                    output.dimensions = input.dimensions;
                    output.type       = input.type;
                    ::memcpy(output.dim, input.dim, input.dimensions * sizeof(halide_dimension_t));
                    output.dim[axis].extent = splitDim;
                }
            } else {
                // one dimension tensor, ex: [5,30]=>[5,4]+[5,15]+[5,11], slicePoints is [4, 15, 11]
                MNN_ASSERT(slice->slicePoints()->size() == outputs.size());
                int determineTensorIndex = -1;
                int maxSize              = 0;
                for (int i = 0; i < slice->slicePoints()->size(); i++) {
                    auto& output      = outputs[i]->buffer();
                    output.type       = input.type;
                    output.dimensions = input.dimensions;
                    ::memcpy(output.dim, input.dim, input.dimensions * sizeof(halide_dimension_t));
                    auto length = slice->slicePoints()->data()[i];
                    if (-1 != length) {
                        output.dim[axis].extent = length;
                        maxSize += length;
                    } else {
                        if (determineTensorIndex >= 0) {
                            // Don't support two -1 points
                            return false;
                        }
                        determineTensorIndex = i;
                    }
                }
                if (determineTensorIndex >= 0) {
                    auto& output            = outputs[determineTensorIndex]->buffer();
                    output.dim[axis].extent = input.dim[axis].extent - maxSize;
                }
            }
        }
        return true;
    }
};

REGISTER_SHAPE(SliceComputer, OpType_Slice);
} // namespace MNN
