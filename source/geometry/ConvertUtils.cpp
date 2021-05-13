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
    auto inputSlice = inputDes->regions;
    MNN_ASSERT(input->dimensions() >= 1);
    MNN_ASSERT(output->dimensions() == input->dimensions());
    if (inputSlice.empty()) {
        inputSlice.resize(1);
        // Create Full Refence
        inputSlice[0] = TensorUtils::makeFullSlice(input);
    }
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

void ConvertUtils::broadcastto(Tensor* input, Tensor* output) {
    auto inputDes         = TensorUtils::getDescribe(input);
    auto outputDes        = TensorUtils::getDescribe(output);
    outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
    if (input->elementSize() == output->elementSize()) {
        // Just Copy Tensor
        auto inputSlice = inputDes->regions;
        if (inputSlice.empty()) {
            // Create Full Refence
            Tensor::InsideDescribe::Region totalSlice = TensorUtils::makeFullSlice(input);
            inputSlice.emplace_back(std::move(totalSlice));
        }
        outputDes->regions = std::move(inputSlice);
        return;
    }
    int32_t inputShape[MNN_MAX_TENSOR_DIM];
    auto outputDim = output->dimensions();
    for (int i=0; i<outputDim; ++i) {
        inputShape[i] = 1;
    }
    int offset = outputDim - input->dimensions();
    for (int i = 0; i < input->dimensions(); ++i) {
        inputShape[i + offset] = input->length(i);
    }
    // Compute Strides
    int sepInputShapeSize = 0;
    int sepOutputShapeSize = 0;
    int sepInputShape[MNN_MAX_TENSOR_DIM];
    int sepOutputShape[MNN_MAX_TENSOR_DIM];
    int currentInput  = 1;
    int currentOutput = 1;
    for (int i = 0; i < outputDim; ++i) {
        if (inputShape[i] != output->length(i)) {
            if (1 < currentOutput) {
                sepInputShape[sepInputShapeSize++] = currentInput;
                sepOutputShape[sepOutputShapeSize++] = currentOutput;
            }
            sepInputShape[sepInputShapeSize++] = (inputShape[i]);
            sepOutputShape[sepOutputShapeSize++] = (output->length(i));
            currentInput  = 1;
            currentOutput = 1;
        } else {
            currentInput *= inputShape[i];
            currentOutput *= output->length(i);
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
    std::vector<int> remainStride(remainDimSize + 1);
    int remainSize = OpCommonUtils::computeStride(remainStride.data(), sepOutputShape, remainDimSize);
    outputDes->regions.resize(remainSize);
    std::vector<int> cords(remainDimSize + 1);
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
