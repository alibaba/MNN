//
//  ShapeSlice.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include <algorithm>
#include <numeric>
namespace MNN {

class SliceComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        //MNN_ASSERT(1 == inputs.size());
        auto outputSize = (int)outputs.size();
        auto slice = op->main_as_Slice();

        auto& input = inputs[0]->buffer();

        int axis = slice->axis();
        if (axis < 0) {
            axis += input.dimensions;
        }

        /*
         If we want split (2, 10) => (2, 3) + (2, 5) + (2, 2), slicePoints is
         1. [3, 8, 10] when slice->sourceType = NetSource_CAFFE
         2. [3, 5, 2] otherwise
         */
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
            output.dimensions = input.dimensions;
            output.type             = input.type;
            ::memcpy(output.dim, input.dim, input.dimensions * sizeof(halide_dimension_t));

            output.dim[axis].extent = input.dim[axis].extent - previous;
        } else {
            // tensorflow/Torch Split
            if (inputs.size() == 1 && (nullptr == slice->slicePoints() || 1 == slice->slicePoints()->size())) {
                // slicePoint size is 1:
                // TF value is num_split, Torch value is split_size
                int numSplits = outputSize,
                    splitDim = input.dim[axis].extent / numSplits;
                if (MNN::NetSource_TORCH == slice->sourceType()) {
                    if (nullptr != slice->slicePoints()) {
                        splitDim = slice->slicePoints()->data()[0];
                    }
                    numSplits = input.dim[axis].extent / splitDim;
                } else if (MNN::NetSource_TENSORFLOW == slice->sourceType()) {
                    if (nullptr != slice->slicePoints() && slice->slicePoints()->data()[0] != outputSize) {
                        numSplits = slice->slicePoints()->data()[0];
                    }
                    MNN_ASSERT(0 == input.dim[axis].extent % numSplits);
                    splitDim = input.dim[axis].extent / numSplits;
                }
                for (int i = 0; i < outputSize; i++) {
                    auto& output      = outputs[i]->buffer();
                    output.dimensions = input.dimensions;
                    output.type       = input.type;
                    ::memcpy(output.dim, input.dim, input.dimensions * sizeof(halide_dimension_t));
                    output.dim[axis].extent = splitDim;
                }
            } else {
                std::vector<int> slicePoints;
                if (inputs.size() == 2) {
                    slicePoints.assign(inputs[1]->host<int>(), inputs[1]->host<int>() + inputs[1]->elementSize());
                } else if (slice->slicePoints() != nullptr) {
                    slicePoints.assign(slice->slicePoints()->begin(), slice->slicePoints()->end());
                }
                int totalLen = std::accumulate(slicePoints.begin(), slicePoints.end(), 0);
                if (totalLen > inputs[0]->length(axis)) {
                    MNN_ASSERT(false);
                    return false;
                }
                int numberSplits = slicePoints.size();
                MNN_ASSERT(0 < numberSplits);
                numberSplits = std::min(numberSplits, outputSize);
                int determineTensorIndex = -1;
                int maxSize              = 0;
                for (int i = 0; i < numberSplits; i++) {
                    auto& output      = outputs[i]->buffer();
                    output.type       = input.type;
                    output.dimensions = input.dimensions;
                    ::memcpy(output.dim, input.dim, input.dimensions * sizeof(halide_dimension_t));
                    auto length = slicePoints[i];
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
        for (int i=0; i<outputs.size(); ++i) {
            TensorUtils::getDescribe(outputs[i])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        }
        return true;
    }
};

REGISTER_SHAPE_INPUTS(SliceComputer, OpType_Slice, {1});
} // namespace MNN
