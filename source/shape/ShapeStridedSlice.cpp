//
//  ShapeStridedSlice.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <array>
#include "CPUStridedSlice.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {
class StridedSliceComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(4 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        const std::string name = op->name()->c_str();

        Tensor *input            = inputs[0];
        const int inputDimension = input->buffer().dimensions;
        MNN_ASSERT(inputDimension > 0);
        if (inputDimension >= 5) {
            MNN_ERROR("Error for StridedSliceComputer: inputDimension>=5: %d\n", inputDimension);
            return false;
        }

        // input haven't realized
        auto output    = outputs[0];
        auto parameter = op->main_as_StridedSliceParam();

        Tensor *begin   = inputs[1];
        Tensor *end     = inputs[2];
        Tensor *strided = inputs[3];

        std::shared_ptr<Tensor> tempBegin;
        std::shared_ptr<Tensor> tempEnd;
        std::shared_ptr<Tensor> tempStrided;

        // copy data from device to host if needed
        if (!begin->host<int32_t>() && begin->deviceId()) {
            tempBegin.reset(Tensor::createHostTensorFromDevice(begin, true));
            begin = tempBegin.get();
        }
        if (!end->host<int32_t>() && end->deviceId()) {
            tempEnd.reset(Tensor::createHostTensorFromDevice(end, true));
            end = tempEnd.get();
        }
        if (!strided->host<int32_t>() && strided->deviceId()) {
            tempStrided.reset(Tensor::createHostTensorFromDevice(strided, true));
            strided = tempStrided.get();
        }

        MNN_ASSERT(begin->buffer().dimensions == end->buffer().dimensions &&
                   begin->buffer().dimensions == strided->buffer().dimensions);

        std::vector<int32_t> inputShape(input->buffer().dimensions);
        for (int i = 0; i < input->buffer().dimensions; i++) {
            inputShape[i] = input->buffer().dim[i].extent;
        }

        int stridedSliceDimension = begin->buffer().dim[0].extent;

        std::vector<int32_t> beginShape(stridedSliceDimension);
        std::vector<int32_t> endShape(stridedSliceDimension);
        std::vector<int32_t> stridedShape(stridedSliceDimension);
        std::vector<int32_t> outputShape;
        std::vector<int32_t> outputShapeShrinked;

        std::vector<int32_t> beginMask(stridedSliceDimension);
        for (int i = 0; i < stridedSliceDimension; i++) {
            beginMask[i] = parameter->beginMask() & (1 << i);
        }

        std::vector<int32_t> endMask(stridedSliceDimension);
        for (int i = 0; i < stridedSliceDimension; i++) {
            endMask[i] = parameter->endMask() & (1 << i);
        }

        std::vector<int32_t> shrinkAxisMask(stridedSliceDimension);
        for (int i = 0; i < stridedSliceDimension; i++) {
            shrinkAxisMask[i] = parameter->shrinkAxisMask() & (1 << i);
        }

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

        if (parameter->ellipsisMask() != 0 || parameter->newAxisMask() != 0) {
            MNN_ASSERT(false); // TODO: do not support these two mask now
        }

        for (int i = 0; i < stridedSliceDimension; i++) {
            if (beginMask[i] > 0) {
                beginShape[i] = 0;
            } else {
                beginShape[i] = std::min(inputShape[i], begin->host<int32_t>()[i]);
            }
            if (beginShape[i] < 0) {
                beginShape[i] += input->buffer().dim[i].extent;
            }
            MNN_ASSERT(beginShape[i] >= 0);
            endShape[i] = endMask[i] > 0
                              ? inputShape[i]
                              : (end->host<int32_t>()[i] > inputShape[i] ? inputShape[i] : end->host<int32_t>()[i]);
            if (endShape[i] < 0) {
                endShape[i] += input->buffer().dim[i].extent;
            }
            MNN_ASSERT(endShape[i] >= 0);
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
                outputShape.push_back(size);
                outputShapeShrinked.push_back(size);
            } else {
                outputShape.push_back(1);
            }
        }

        int outputDimensionsWithoutRemain = (int)outputShape.size();
        int dimensionRemained             = input->buffer().dimensions - stridedSliceDimension;

        for (int i = 0; i < dimensionRemained; i++) {
            outputShape.push_back(input->buffer().dim[outputDimensionsWithoutRemain + i].extent);
            outputShapeShrinked.push_back(input->buffer().dim[outputDimensionsWithoutRemain + i].extent);
        }

        output->buffer().dimensions    = (int)outputShapeShrinked.size();
        output->buffer().type          = input->buffer().type;
        output->buffer().dim[0].extent = 1;

        for (int i = 0; i < outputShapeShrinked.size(); i++) {
            output->buffer().dim[i].extent = outputShapeShrinked[i];
            output->buffer().dim[i].flags  = 0;
        }

        return true;
    }
};

REGISTER_SHAPE(StridedSliceComputer, OpType_StridedSlice);
} // namespace MNN
