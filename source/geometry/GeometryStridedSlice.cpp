//
//  GeometryStridedSlice.cpp
//  MNN
//
//  Created by MNN on 2020/04/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/Macro.h"
#include "ConvertUtils.hpp"
namespace MNN {
class GeometryStridedSlice : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        Tensor* input = inputs[0];
        // input haven't realized
        auto output     = outputs[0];
        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        const int inputDim = input->buffer().dimensions;
        auto parameter = op->main_as_StridedSliceParam();
        int32_t beginMask = parameter->beginMask();
        int32_t endMask = parameter->endMask();
        int32_t shrinkAxisMask = parameter->shrinkAxisMask();
        int32_t ellipsisMask = parameter->ellipsisMask();
        int32_t newAxisMask = parameter->newAxisMask();
        int32_t fromType = parameter->fromType();
        if (ellipsisMask && (ellipsisMask & (ellipsisMask - 1))) {
            MNN_ERROR("only one non-zero bit is allowed in ellipsisMask\n");
            return false;
        }

        MNN_ASSERT(inputs.size() >= 3 && inputs.size() <= 5);
        Tensor *begin   = inputs[1];
        Tensor *end     = inputs[2];

        int32_t strideSize = begin->length(0);
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
        int32_t inputStride[MNN_MAX_TENSOR_DIM];
        
        {
            int stride = 1;
            for (int i = input->buffer().dimensions - 1; i >= 0; --i) {
                inputShape[i]  = input->buffer().dim[i].extent;
                inputStride[i] = stride;
                stride *= inputShape[i];
            }
        }
        
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
        
        int32_t beginShape[MNN_MAX_TENSOR_DIM] = { 0 };
        int32_t endShape[MNN_MAX_TENSOR_DIM] = { 0 };
        int32_t stridedShape[MNN_MAX_TENSOR_DIM] = { 0 };
        int32_t outputShape[MNN_MAX_TENSOR_DIM] = { 0 };
        int32_t reverseDim = -1;
        int32_t shapeNum = 0;

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

        for (int i = 0; i < strideSize; i++) {
            if (newAxisMasks[i] > 0) {
                // ignore newAxis beacuse it is 1
                continue;
            }
            stridedShape[shapeNum] = (shrinkAxisMasks[i] > 0 ? 1 : strides[i]);
            if (stridedShape[shapeNum] < 0) {
                reverseDim = i;
            }
            if (beginMasks[i] > 0) {
                beginShape[shapeNum] = stridedShape[shapeNum] < 0 ? inputShape[shapeNum] - 1 : 0;
            } else {
                beginShape[shapeNum] = stridedShape[shapeNum] < 0 ? beginAndEndShapeLimit(begins[i], inputShape[shapeNum], false) :
                                                                    std::min(inputShape[shapeNum], begins[i]);
            }
            if (beginShape[shapeNum] < 0) {
                auto temp = -beginShape[shapeNum];
                beginShape[shapeNum] = UP_DIV(temp, input->buffer().dim[i].extent) * input->buffer().dim[i].extent + beginShape[shapeNum];
            }
            if (endMasks[i] > 0) {
                endShape[shapeNum] = stridedShape[shapeNum] < 0 ? -1 : inputShape[shapeNum];
            } else {
                endShape[shapeNum] = stridedShape[shapeNum] < 0 ? std::max(-1, std::min(inputDim, ends[i])) :
                                                                  beginAndEndShapeLimit(ends[i], inputShape[shapeNum], true);
            }

            if (shrinkAxisMasks[i] == 0) {
                if (stridedShape[shapeNum] > 0) {
                    int size = (endShape[shapeNum] - beginShape[shapeNum] - 1) / stridedShape[shapeNum] + 1;
                    outputShape[shapeNum] = size;
                } else {
                    int size = (endShape[shapeNum] - beginShape[shapeNum] + 1) / stridedShape[shapeNum] + 1;
                    outputShape[shapeNum] = size;
                }
            } else {
                outputShape[shapeNum] = 1;
            }
            shapeNum++;
        }
        int dealDims = shapeNum;
        int dimensionRemained = input->dimensions() - dealDims;
        for (int i = 0; i < dimensionRemained; i++) {
            outputShape[shapeNum] = input->length(dealDims + i);
            stridedShape[shapeNum] = 1;
            beginShape[shapeNum] = 0;
            shapeNum++;
        }
        int remainSize = 1;
        int remainDims[MNN_MAX_TENSOR_DIM];
        int remainDimSize = shapeNum - 3;
        for (int i = 0; i < (int)shapeNum - 3; ++i) {
            remainSize *= outputShape[i];
            remainDims[i] = outputShape[i];
        }
        outputDes->regions.resize(remainSize);
        int regionSize        = shapeNum < 3 ? shapeNum : 3;
        if (reverseDim >= 0) {
            remainDimSize = reverseDim;
            for (int i = 0; i < reverseDim; ++i) {
                remainSize *= outputShape[i];
                remainDims[i] = outputShape[i];
            }
            outputDes->regions.resize(remainSize);
            regionSize = shapeNum - reverseDim;
            MNN_ASSERT(regionSize <= 3);
        }
        int mod[MNN_MAX_TENSOR_DIM];
        OpCommonUtils::computeStride(mod, remainDims, (int)remainDimSize);
        int outputStrideTotal = 1;
        int basicInputOffset  = 0;
        for (int i = 0; i < shapeNum - regionSize; ++i) {
            basicInputOffset += inputStride[i] * beginShape[i];
        }
        for (int i = 0; i < regionSize; ++i) {
            int pos  = shapeNum - i - 1;
            auto len = outputShape[pos];
            basicInputOffset += inputStride[pos] * beginShape[pos];
            outputStrideTotal *= len;
        }
        int coordinates[MNN_MAX_TENSOR_DIM];
        for (int r = 0; r < remainSize; ++r) {
            OpCommonUtils::unravelIndexHelper(coordinates, mod, remainDimSize, r);
            int inputOffset = basicInputOffset;
            for (int i = 0; i < remainDimSize; ++i) {
                inputOffset += coordinates[i] * inputStride[i] * stridedShape[i];
            }

            auto& reg      = outputDes->regions[r];
            reg.dst.offset = r * outputStrideTotal;
            reg.src.offset = inputOffset;
            reg.origin     = input;
            for (int i = 0; i < regionSize; ++i) {
                int pos                   = shapeNum - i - 1;
                reg.size[3 - i - 1]       = outputShape[pos];
                reg.src.stride[3 - i - 1] = inputStride[pos] * stridedShape[pos];
            }
            reg.dst.stride[0] = reg.size[1] * reg.size[2];
            reg.dst.stride[1] = reg.size[2];
            reg.dst.stride[2] = 1;
        }

        if (fromType == 0 && inputs.size() == 5) {
            auto write = inputs[4];
            std::vector<int> shape(outputShape, outputShape + shapeNum);
            if (write->shape() != shape) {
                std::shared_ptr<Tensor> newTensor(new Tensor);
                newTensor->buffer().type = write->buffer().type;
                newTensor->buffer().dimensions = shapeNum;
                for (int i = 0; i < shapeNum; i++) {
                    newTensor->setLength(i, outputShape[i]);
                }
                ConvertUtils::broadcastto(write, newTensor.get());
                write = newTensor.get();
                res.extras.emplace_back(newTensor);
            }
            for (auto& reg : outputDes->regions) {
                auto tmp = reg.dst;
                reg.dst = reg.src;
                reg.src = tmp;
                reg.origin = write;
            }
            Tensor::InsideDescribe::Region region;
            region.size[2] = input->elementSize();
            region.origin = input;
            outputDes->regions.insert(outputDes->regions.begin(), region);
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryStridedSlice);
    GeometryComputer::registerGeometryComputer(comp, {OpType_StridedSlice});
}

REGISTER_GEOMETRY(GeometryStridedSlice, _create);

} // namespace MNN
