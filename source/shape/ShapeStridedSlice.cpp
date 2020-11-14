//
//  ShapeStridedSlice.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <array>
#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
namespace MNN {
class StridedSliceComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(4 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        
        Tensor *input            = inputs[0];
        const int inputDimension = input->buffer().dimensions;
        if (inputDimension <= 0) {
            return false;
        }
        // input haven't realized
        auto output    = outputs[0];
        auto parameter = op->main_as_StridedSliceParam();

        Tensor *begin   = inputs[1];
        Tensor *end     = inputs[2];
        Tensor *strided = inputs[3];

        MNN_ASSERT(begin->buffer().dimensions == end->buffer().dimensions &&
                   begin->buffer().dimensions == strided->buffer().dimensions);

        int32_t inputShape[MNN_MAX_TENSOR_DIM];
        for (int i = 0; i < input->buffer().dimensions; i++) {
            inputShape[i] = input->buffer().dim[i].extent;
        }

        int stridedSliceDimension = begin->buffer().dim[0].extent;

        int32_t beginShape[MNN_MAX_TENSOR_DIM];
        int32_t endShape[MNN_MAX_TENSOR_DIM];
        int32_t stridedShape[MNN_MAX_TENSOR_DIM];
        int32_t outputShape[MNN_MAX_TENSOR_DIM];
        int32_t outputShapeShrinked[MNN_MAX_TENSOR_DIM];
        int outputShapeSize = 0;
        int outputShapeShrinkSize = 0;

        int32_t beginMask[MNN_MAX_TENSOR_DIM];
        for (int i = 0; i < stridedSliceDimension; i++) {
            beginMask[i] = parameter->beginMask() & (1 << i);
        }

        int32_t endMask[MNN_MAX_TENSOR_DIM];
        for (int i = 0; i < stridedSliceDimension; i++) {
            endMask[i] = parameter->endMask() & (1 << i);
        }

        int32_t shrinkAxisMask[MNN_MAX_TENSOR_DIM];
        for (int i = 0; i < stridedSliceDimension; i++) {
            shrinkAxisMask[i] = parameter->shrinkAxisMask() & (1 << i);
        }
#ifdef MNN_SUPPORT_ELLIPSE
        int ellipsisMaskNonZeroBitPosition = 0;
        for (int i = 0; i < stridedSliceDimension; i++) {
            int temp = parameter->ellipsisMask() & (1 << i);
            if (temp != 0) {
                ellipsisMaskNonZeroBitPosition = i; // only one non-zero bit is allowed in ellipsisMask
                break;
            }
        }

        std::vector<int32_t> newAxisMask(stridedSliceDimension);
        for (int i = 0; i < stridedSliceDimension; i++) {
            newAxisMask[i] = parameter->newAxisMask() & (1 << i);
        }
#endif
        if (parameter->ellipsisMask() != 0 || parameter->newAxisMask() != 0) {
            MNN_ERROR("Strided_slice don't support ellipsisMask and newAxisMask now\n");
            return false;
        }

        auto beginAndEndShapeLimit = [](int shape, int dimSize, bool exclusive) -> int {
            int maxShape = dimSize - 1, minShape = -dimSize;
            if (exclusive) {
                ++maxShape;
                --minShape;
            }
            shape = (shape > maxShape ? maxShape : shape);
            shape = (shape < minShape ? minShape : shape);
            if (shape < 0) {
                shape += dimSize;
            }
            return shape;
        };

        for (int i = 0; i < stridedSliceDimension; i++) {
            if (beginMask[i] > 0) {
                beginShape[i] = 0;
            } else {
                beginShape[i] = std::min(inputShape[i], begin->host<int32_t>()[i]);
            }
            if (beginShape[i] < 0) {
                beginShape[i] += input->buffer().dim[i].extent;
            }
            if (endMask[i] > 0) {
                endShape[i] = inputShape[i];
            } else {
                endShape[i] = beginAndEndShapeLimit(end->host<int32_t>()[i], inputShape[i], true);
            }
            stridedShape[i] = shrinkAxisMask[i] > 0 ? 1 : strided->host<int32_t>()[i];

            if (endShape[i] < beginShape[i]) {
                int t         = beginShape[i];
                beginShape[i] = endShape[i];
                endShape[i]   = t;

                MNN_ASSERT(stridedShape[i] != 0);
                if (stridedShape[i] < 0) {
                    stridedShape[i] = -stridedShape[i];
                } else {
                    // MNN_ASSERT(false);  // TODO: should be the wrong case, but there is one in linfeng's faster
                    // rcnn face model
                    beginShape[i] = endShape[i]; // TODO: temp solution
                }
            }

            if (shrinkAxisMask[i] == 0) {
                int size = (endShape[i] - beginShape[i] - 1) / stridedShape[i] + 1;
                outputShape[outputShapeSize] = size;
                outputShapeSize++;
                outputShapeShrinked[outputShapeShrinkSize] = size;
                outputShapeShrinkSize++;
            } else {
                outputShape[outputShapeSize] = std::min(1, inputShape[i]);
                outputShapeSize++;
            }
        }

        int outputDimensionsWithoutRemain = outputShapeSize;
        int dimensionRemained             = input->buffer().dimensions - stridedSliceDimension;

        for (int i = 0; i < dimensionRemained; i++) {
            outputShapeShrinked[outputShapeShrinkSize] = input->buffer().dim[outputDimensionsWithoutRemain + i].extent;
            outputShapeShrinkSize++;
        }

        output->buffer().dimensions    = outputShapeShrinkSize;
        output->buffer().type          = input->buffer().type;
        output->buffer().dim[0].extent = 1;

        for (int i = 0; i < outputShapeShrinkSize; i++) {
            output->buffer().dim[i].extent = outputShapeShrinked[i];
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
};

REGISTER_SHAPE_INPUTS(StridedSliceComputer, OpType_StridedSlice, (std::vector<int>{1,2,3}));
} // namespace MNN
