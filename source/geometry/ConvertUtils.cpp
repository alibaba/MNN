//
//  ConvertUtils.cpp
//  MNN
//
//  Created by MNN on 2020/04/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvertUtils.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {
bool ConvertUtils::compute(Tensor* input, Tensor* output, CommandBuffer& res) {
    auto inputDes     = TensorUtils::getDescribe(input);
    auto outputDes    = TensorUtils::getDescribe(output);
    auto inputFormat  = inputDes->dimensionFormat;
    auto outputFormat = outputDes->dimensionFormat;
    if (MNN_DATA_FORMAT_NC4HW4 == inputFormat) {
        inputFormat = MNN_DATA_FORMAT_NCHW;
    }
    if (MNN_DATA_FORMAT_NC4HW4 == outputFormat) {
        outputFormat = MNN_DATA_FORMAT_NCHW;
    }
    std::vector<Tensor::InsideDescribe::Region> inputSlice = {TensorUtils::makeFullSlice(input)};
    if (inputFormat == outputFormat || 2 == input->dimensions()) {
        // No need for treat for NCWH <-> NC4HW4
        outputDes->regions    = std::move(inputSlice);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        return true;
    }
    // NHWC <-> NC4HW4: Turn NHWC to NCHW
    // TODO for multi input can find better way to compute new slice
    MNN_ASSERT(4 == input->dimensions());
    auto inside  = input->width() * input->height();
    auto axis    = input->channel();
    auto outside = input->batch();
    auto swap    = [](Tensor::InsideDescribe::Region& inp) {
        auto tempStride   = inp.src.stride[2];
        inp.src.stride[2] = inp.src.stride[1];
        inp.src.stride[1] = tempStride;
        auto tempSize     = inp.size[2];
        inp.size[2]       = inp.size[1];
        inp.size[1]       = tempSize;
        inp.dst.stride[2] = 1;
        inp.dst.stride[1] = inp.size[2];
    };
    if (inputSlice.size() == 1) {
        auto& inp       = inputSlice[0];
        bool canReshape = false;
        if (inputFormat == MNN_DATA_FORMAT_NCHW) {
            canReshape = TensorUtils::reshapeSlice(inp, outside, inside, axis);
        } else {
            canReshape = TensorUtils::reshapeSlice(inp, outside, axis, inside);
        }
        if (canReshape) {
            swap(inp);
            outputDes->regions    = std::move(inputSlice);
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            return true;
        }
    }
    auto slice = TensorUtils::makeFullSlice(input);
    if (inputFormat == MNN_DATA_FORMAT_NCHW) {
        TensorUtils::reshapeSlice(slice, outside, inside, axis);
    } else {
        TensorUtils::reshapeSlice(slice, outside, axis, inside);
    }
    swap(slice);

    outputDes->regions    = {slice};
    outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;

    return true;
}

void ConvertUtils::broadcastto(Tensor* input, Tensor* output, bool forward) {
    
    auto outputDes        = TensorUtils::getDescribe(output);
    outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
    if (TensorUtils::getRawSize(input) == TensorUtils::getRawSize(output)) {
        // Just Copy Tensor
        outputDes->regions = {TensorUtils::makeFullSlice(input)};
        return;
    }
    // if forward ( tf select broadcast )
    if (forward) {
        MNN_ASSERT(input->dimensions() == 1 && output->dimensions() > 1);
        MNN_ASSERT(input->length(0) == output->length(0));
        int srcSize = input->length(0);
        int multipler = output->length(1);
        for (int i = 2; i < output->dimensions(); i++) {
            multipler *= output->length(i);
        }
        // [srcSize] -> [srcSize, multipler]
        outputDes->regions.resize(1);
        auto& reg = outputDes->regions[0];
        reg.size[0] = 1;
        reg.size[1] = srcSize;
        reg.size[2] = multipler;
        reg.src.offset = 0;
        reg.src.stride[0] = srcSize;
        reg.src.stride[1] = 1;
        reg.src.stride[2] = 0;
        reg.dst.offset = 0;
        reg.dst.stride[0] = srcSize * multipler;
        reg.dst.stride[1] = multipler;
        reg.dst.stride[2] = 1;
        reg.origin = input;
    }
    int32_t inputShape[MNN_MAX_TENSOR_DIM];
    int32_t outputShape[MNN_MAX_TENSOR_DIM];

    auto outputDim = output->dimensions();
    for (int i=0; i<outputDim; ++i) {
        inputShape[i] = 1;
        outputShape[i] = output->length(i);
    }
    int offset = outputDim - input->dimensions();
    for (int i = 0; i < input->dimensions(); ++i) {
        inputShape[i + offset] = input->length(i);
    }

    // Squeeze consecutive 1 dimension
    while(outputDim >= 2) {
        bool canFuse = false;
        for(int i=0; i<outputDim-1; ++i) {
            if(inputShape[i] == 1 && inputShape[i+1] == 1) {
                for(int j=i+1; j<outputDim; j++) {
                    inputShape[j] = inputShape[j+1];
                }
                outputShape[i] *= outputShape[i+1];
                for(int j=i+1; j<outputDim; j++) {
                    outputShape[j] = outputShape[j+1];
                }
                outputDim--;
                i--;
                canFuse = true;
            }
        }
        if(!canFuse) {
            break;
        }
    }

    // Compute Strides
    int sepInputShapeSize = 0;
    int sepOutputShapeSize = 0;
    int sepInputShape[MNN_MAX_TENSOR_DIM];
    int sepOutputShape[MNN_MAX_TENSOR_DIM];
    int currentInput  = 1;
    int currentOutput = 1;
    for (int i = 0; i < outputDim; ++i) {
        if (inputShape[i] != outputShape[i]) {
            if (1 < currentOutput) {
                sepInputShape[sepInputShapeSize++] = currentInput;
                sepOutputShape[sepOutputShapeSize++] = currentOutput;
            }
            sepInputShape[sepInputShapeSize++] = (inputShape[i]);
            sepOutputShape[sepOutputShapeSize++] = (outputShape[i]);
            currentInput  = 1;
            currentOutput = 1;
        } else {
            currentInput *= inputShape[i];
            currentOutput *= outputShape[i];
        }
    }
    if (currentOutput != 1 || currentInput != 1) {
        sepInputShape[sepInputShapeSize++] = (currentInput);
        sepOutputShape[sepOutputShapeSize++] = (currentOutput);
    }
    int seperateOutputStrides[MNN_MAX_TENSOR_DIM];
    int seperateInputStrides[MNN_MAX_TENSOR_DIM];
    OpCommonUtils::computeStride(seperateOutputStrides, sepOutputShape, sepOutputShapeSize);
    OpCommonUtils::computeStride(seperateInputStrides, sepInputShape, sepInputShapeSize);
    for (int i = 0; i < sepInputShapeSize; ++i) {
        if (1 == sepInputShape[i]) {
            seperateInputStrides[i] = 0;
        }
    }

    // Split region by size, use stride to determine src and dst mapping
    int remainDimSize = sepInputShapeSize > 3 ? (int)sepInputShapeSize - 3 : 0;
    int remainStride[MNN_MAX_TENSOR_DIM];
    int remainSize = OpCommonUtils::computeStride(remainStride, sepOutputShape, remainDimSize);
    outputDes->regions.clear();
    outputDes->regions.resize(remainSize);
    int cords[MNN_MAX_TENSOR_DIM];
    for (int index = 0; index < remainSize; ++index) {
        OpCommonUtils::unravelIndexHelper(cords, remainStride, remainDimSize, index);
        auto& reg = outputDes->regions[index];
        for (int i = 0; i < remainDimSize; ++i) {
            reg.src.offset += (cords[i] * seperateInputStrides[i]);
            reg.dst.offset += (cords[i] * seperateOutputStrides[i]);
        }
        reg.origin = input;
        for (int i = 0; i < 3; ++i) {
            auto match = (int)sepOutputShapeSize - i - 1;
            if (match < 0) {
                continue;
            }
            reg.size[3 - i - 1]       = sepOutputShape[match];
            reg.src.stride[3 - i - 1] = seperateInputStrides[match];
            reg.dst.stride[3 - i - 1] = seperateOutputStrides[match];
        }

    }
}

} // namespace MNN
