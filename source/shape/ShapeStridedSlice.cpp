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
        MNN_ASSERT(3 <= inputs.size());
        MNN_ASSERT(5 >= inputs.size());
        MNN_ASSERT(1 == outputs.size());
        
        Tensor *input            = inputs[0];
        const int inputDim = input->buffer().dimensions;
        if (inputDim <= 0 || inputDim > MNN_MAX_TENSOR_DIM) {
            return false;
        }
        auto parameter = op->main_as_StridedSliceParam();
        int32_t beginMask = parameter->beginMask();
        int32_t endMask = parameter->endMask();
        int32_t shrinkAxisMask = parameter->shrinkAxisMask();
        int32_t ellipsisMask = parameter->ellipsisMask();
        int32_t newAxisMask = parameter->newAxisMask();
        int32_t fromType = parameter->fromType();

        // write to input
        if (fromType == 0 && inputs.size() == 5) {
            TensorUtils::copyShape(inputs[0], outputs[0], true);
            outputs[0]->buffer().type = inputs[0]->buffer().type;
            TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
            return true;
        }
        if (ellipsisMask && (ellipsisMask & (ellipsisMask - 1))) {
            MNN_ERROR("only one non-zero bit is allowed in ellipsisMask\n");
            return false;
        }

        Tensor *begin   = inputs[1];
        Tensor *end     = inputs[2];

        int32_t strideSize = begin->length(0);
        auto output    = outputs[0];

        MNN_ASSERT(begin->buffer().dimensions == end->buffer().dimensions);
        int32_t inputShape[MNN_MAX_TENSOR_DIM] = { 0 };
        int32_t begins[MNN_MAX_TENSOR_DIM] = { 0 };
        int32_t ends[MNN_MAX_TENSOR_DIM] = { 0 };
        int32_t strides[MNN_MAX_TENSOR_DIM] = { 0 };
        int32_t axes[MNN_MAX_TENSOR_DIM] = { 0 };
        int32_t beginMasks[MNN_MAX_TENSOR_DIM] = { 0 };
        int32_t endMasks[MNN_MAX_TENSOR_DIM] = { 0 };
        int32_t shrinkAxisMasks[MNN_MAX_TENSOR_DIM] = { 0 };
        int32_t newAxisMasks[MNN_MAX_TENSOR_DIM] = { 0 };

        for (int i = 0; i < inputDim; i++) {
            inputShape[i] = input->length(i);
        }

        for (int i = 0; i < strideSize; i++) {
            beginMasks[i] = beginMask & (1 << i);
        }
        for (int i = 0; i < strideSize; i++) {
            endMasks[i] = endMask & (1 << i);
        }
        for (int i = 0; i < strideSize; i++) {
            shrinkAxisMasks[i] = shrinkAxisMask & (1 << i);
        }
        for (int i = 0; i < strideSize; i++) {
            newAxisMasks[i] = newAxisMask & (1 << i);
        }

        for(int i = 0; i < inputDim; i++) {
            begins[i] = 0;
            ends[i] = inputShape[i];
            strides[i] = 1;
        }
        
        // broadcast begin end stride axis param
        if (fromType == 1) {

            Tensor *axis = nullptr;
            if(inputs.size() >= 4) {
                axis = inputs[3];
            }

            Tensor *step = nullptr;
            if(inputs.size() == 5) {
                step = inputs[4];
            }

            for(int i = 0; i < inputDim; i++) {
                begins[i] = 0;
                ends[i] = inputShape[i];
                strides[i] = 1;
            }
        
            for (int i = 0; i < strideSize; i++) {
                auto temp_axis = i;
                if(axis != nullptr) {
                    temp_axis = axis->host<int>()[i];
                    temp_axis = temp_axis < 0 ? (temp_axis + inputDim) : temp_axis;
                    MNN_ASSERT(temp_axis < MNN_MAX_TENSOR_DIM);
                }
                if(step != nullptr) {
                    strides[temp_axis] = step->host<int>()[i];
                }
                
                auto shape = inputShape[temp_axis];
                auto temp_value = begin->host<int>()[i];
                temp_value = temp_value < 0 ? (temp_value + shape) : temp_value;
                begins[temp_axis] = temp_value;
                
                temp_value = end->host<int>()[i];
                temp_value = temp_value < 0 ? (temp_value + shape) : temp_value;
                ends[temp_axis] = temp_value;
            }
            strideSize = inputDim;

        } else if(fromType == 0) {

            Tensor *strided = nullptr;
            if(inputs.size() >= 4) {
                strided = inputs[3];
                MNN_ASSERT(begin->buffer().dimensions == strided->buffer().dimensions);
            }
            // deal ellipsis, expand strides info
            if (ellipsisMask > 0) {
                int32_t beginMasksTmp[MNN_MAX_TENSOR_DIM] = { 0 };
                int32_t endMasksTmp[MNN_MAX_TENSOR_DIM] = { 0 };
                int32_t shrinkAxisMasksTmp[MNN_MAX_TENSOR_DIM] = { 0 };
                int32_t newAxisMasksTmp[MNN_MAX_TENSOR_DIM] = { 0 };
                // expand stride info
                int ellipsisPos = -1;
                for (int i = 0; i < strideSize; i++) {
                    int temp = ellipsisMask & (1 << i);
                    if (temp != 0) {
                        ellipsisPos = i;
                        break;
                    }
                }
                MNN_ASSERT(ellipsisPos >= 0 && ellipsisPos < strideSize);
                /*
                Example: foo's dim is [2, 3, 4, 5, 6, 7], foo[0:2, :, 3:5, 3:6]:
                    1. strideSize = 4, inputDim = 6, ellipsis = 2(0010)
                    2. left part: 0:2, right part: 3:5, 3:6
                    3. expand: foo[0:2, 0:3, 0:4, 3:5, 3:6]
                */
                int ellpsisSize = inputDim - strideSize, strideIdx = 0;
                for (int i = 0; i < inputDim; i++) {
                    if (i == ellipsisPos) {
                        strideIdx++;
                    }
                    if (i >= ellipsisPos && i <= ellipsisPos + ellpsisSize) {
                        begins[i] = 0;
                        ends[i] = inputShape[i];
                        strides[i] = 1;
                        beginMasksTmp[i] = 0;
                        endMasksTmp[i] = 0;
                        shrinkAxisMasksTmp[i] = 0;
                    } else {
                        begins[i] = begin->host<int32_t>()[strideIdx];
                        ends[i] = end->host<int32_t>()[strideIdx];
                        if(strided != nullptr) {
                            strides[i] = strided->host<int32_t>()[strideIdx];
                        }
                        beginMasksTmp[i] = beginMasks[strideIdx];
                        endMasksTmp[i] = endMasks[strideIdx];
                        shrinkAxisMasksTmp[i] = shrinkAxisMasks[strideIdx];
                        newAxisMasksTmp[i] = newAxisMasks[strideIdx++];
                    }
                }
                for (int i = 0; i < inputDim; i++) {
                    beginMasks[i] = beginMasksTmp[i];
                    endMasks[i] = endMasksTmp[i];
                    shrinkAxisMasks[i] = shrinkAxisMasksTmp[i];
                    newAxisMasks[i] = newAxisMasksTmp[i];
                }
                strideSize = inputDim;
            } else {
                for (int i = 0; i < strideSize; i++) {
                    begins[i] = begin->host<int>()[i];
                    ends[i] = end->host<int>()[i];
                    strides[i] = strided->host<int>()[i];
                }
            }
        }

        int32_t beginShape[MNN_MAX_TENSOR_DIM];
        int32_t endShape[MNN_MAX_TENSOR_DIM];
        int32_t stridedShape[MNN_MAX_TENSOR_DIM];
        int32_t outputShape[MNN_MAX_TENSOR_DIM];
        int32_t outputShapeShrinked[MNN_MAX_TENSOR_DIM];

        int outputShapeSize = 0;
        int outputShapeShrinkSize = 0;
        int strideDealDims = 0;

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

        int inputDimOffset = 0;
        for (int i = 0; i < strideSize; i++) {
            if (newAxisMasks[i] > 0) {
                outputShape[outputShapeSize] = 1;
                outputShapeSize++;
                outputShapeShrinked[outputShapeShrinkSize] = 1;
                outputShapeShrinkSize++;
                continue;
            }
            auto inputDim = inputShape[inputDimOffset++];
            strideDealDims++;
            stridedShape[i] = shrinkAxisMasks[i] > 0 ? 1 : strides[i];
            if (beginMasks[i] > 0) {
                beginShape[i] = stridedShape[i] < 0 ? inputDim - 1 : 0;
            } else {
                beginShape[i] = stridedShape[i] < 0 ? beginAndEndShapeLimit(begins[i], inputDim, false) :
                                                      std::min(inputDim, begins[i]);
            }
            if (beginShape[i] < 0) {
                auto temp = -beginShape[i];
                beginShape[i] = UP_DIV(temp, inputDim) * inputDim + beginShape[i];
            }
            if (endMasks[i] > 0) {
                endShape[i] = stridedShape[i] < 0 ? -1 : inputDim;
            } else {
                endShape[i] = stridedShape[i] < 0 ? std::max(-1, std::min(inputDim, ends[i])) :
                                                    beginAndEndShapeLimit(ends[i], inputDim, true);
            }

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

            if (shrinkAxisMasks[i] == 0) {
                int size = (endShape[i] - beginShape[i] - 1) / stridedShape[i] + 1;
                outputShape[outputShapeSize] = size;
                outputShapeSize++;
                outputShapeShrinked[outputShapeShrinkSize] = size;
                outputShapeShrinkSize++;
            } else {
                outputShape[outputShapeSize] = std::min(1, inputDim);
                outputShapeSize++;
            }
        }

        int outputDimensionsWithoutRemain = strideDealDims;
        int dimensionRemained             = input->buffer().dimensions - strideDealDims;

        for (int i = 0; i < dimensionRemained; i++) {
            outputShapeShrinked[outputShapeShrinkSize] = input->buffer().dim[outputDimensionsWithoutRemain + i].extent;
            outputShapeShrinkSize++;
        }

        output->buffer().dimensions    = outputShapeShrinkSize;
        output->buffer().type          = input->buffer().type;
        for (int i = 0; i < outputShapeShrinkSize; i++) {
            output->buffer().dim[i].extent = outputShapeShrinked[i];
        }

        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
};

REGISTER_SHAPE_INPUTS(StridedSliceComputer, OpType_StridedSlice, (std::vector<int>{1,2,3,4}));
} // namespace MNN
