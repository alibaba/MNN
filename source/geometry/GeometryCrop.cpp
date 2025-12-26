//
//  GeometryCrop.cpp
//  MNN
//
//  Created by MNN on 2020/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {

static int computeOffsetRegion(Tensor::InsideDescribe::NativeInsideDescribe* outputDes, Tensor* input, Tensor* output, Tensor* real,
                               const std::vector<int>& offsets,
                               const std::vector<int>& rightOffsets,
                               std::vector<int>& seperateInputDims,
                               std::vector<int>& seperateOutputDims, std::vector<int>& seperateOffsets,
                               std::vector<int>& seperateInputStrides, std::vector<int>& seperateOutputStrides,
                               int* remainStride,
                               int& remainStrideSize
                               ) {
    int currentInput  = 1;
    int currentOutput = 1;
    int currentSize   = 1;
    auto inputDim     = input->dimensions();
    std::vector<int> seperateRightOffset;
    for (int i = 0; i < inputDim; ++i) {
        if (output->length(i) != input->length(i)) {
            if (1 < currentInput) {
                seperateInputDims.emplace_back(currentInput);
                seperateOutputDims.emplace_back(currentOutput);
                seperateOffsets.emplace_back(0);
                seperateRightOffset.emplace_back(0);
            }
            seperateInputDims.emplace_back(input->length(i));
            seperateOutputDims.emplace_back(output->length(i));
            seperateOffsets.emplace_back(offsets[i]);
            seperateRightOffset.emplace_back(rightOffsets[i]);
            currentInput  = 1;
            currentOutput = 1;
            currentSize = 1;
        } else {
            currentInput *= input->length(i);
            currentOutput *= output->length(i);
        }
    }
    if (currentOutput != 1 || currentInput != 1) {
        seperateInputDims.emplace_back(currentInput);
        seperateOutputDims.emplace_back(currentOutput);
        seperateOffsets.emplace_back(0);
        seperateRightOffset.emplace_back(0);
    }
    seperateOutputStrides.resize(seperateOutputDims.size());
    seperateInputStrides.resize(seperateOutputDims.size());
    OpCommonUtils::computeStride(seperateOutputStrides.data(), seperateOutputDims.data(), seperateOutputDims.size());
    OpCommonUtils::computeStride(seperateInputStrides.data(), seperateInputDims.data(), seperateInputDims.size());

    int remainDimSize = seperateOffsets.size() > 3 ? (int)seperateOffsets.size() - 3 : 0;
    remainStrideSize = remainDimSize;
    int remainSize = OpCommonUtils::computeStride(remainStride, seperateOutputDims.data(), remainDimSize);
    outputDes->regions.clear();
    outputDes->regions.reserve(remainSize);
    outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;

    int cords[MNN_MAX_TENSOR_DIM];
    for (int index = 0; index < remainSize; ++index) {
        OpCommonUtils::unravelIndexHelper(cords, remainStride, remainDimSize, index);
        bool valid = true;
        for (int i = 0; i < remainDimSize; ++i) {
            if (seperateOffsets[i] + cords[i] < 0) {
                valid = false;
                break;
            }
            if (cords[i] - seperateRightOffset[i] >= seperateOutputDims[i]) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            continue;
        }
        Tensor::InsideDescribe::Region reg;
        reg.src.offset = 0;
        reg.dst.offset = 0;
        for (int i = 0; i < remainDimSize; ++i) {
            reg.src.offset += ((cords[i] + seperateOffsets[i]) * seperateInputStrides[i]);
            reg.dst.offset += (cords[i] * seperateOutputStrides[i]);
        }
        MNN_ASSERT(reg.src.offset >= 0);
        reg.origin = real;
        for (int i = 0; i < 3; ++i) {
            auto match = (int)seperateOffsets.size() - i - 1;
            if (match < 0) {
                continue;
            }
            int size = seperateOutputDims[match];
            if (seperateOffsets[match] >=0 ) {
                reg.src.offset += seperateOffsets[match] * seperateInputStrides[match];
            } else {
                reg.dst.offset += (-seperateOffsets[match]) * seperateOutputStrides[match];
                size = size + seperateOffsets[match];
            }
            if (seperateRightOffset[match] < 0) {
                size = size + seperateRightOffset[match];
            }
            reg.size[3 - i - 1] = size;
            reg.src.stride[3 - i - 1] = seperateInputStrides[match];
            reg.dst.stride[3 - i - 1] = seperateOutputStrides[match];
        }
        MNN_ASSERT(reg.src.offset >= 0);
        outputDes->regions.emplace_back(std::move(reg));
    }
    return remainSize;
}

class GeometryCrop : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input         = inputs[0];
        auto cropParam     = op->main_as_Crop();
        auto axis          = cropParam->axis();
        int offsetSize     = cropParam->offset()->size();
        auto offsetData    = cropParam->offset()->data();
        const int inputDim = input->buffer().dimensions;
        if (axis < 0) {
            axis = inputDim + axis;
        }
        MNN_ASSERT(inputDim > 0);
        std::vector<int> offsets(inputDim, 0);
        std::vector<int> rightOffset(inputDim, 0);
        for (int i = 0; i < inputDim; ++i) {
            int cropOffset = 0;
            if (i >= axis) {
                if (offsetSize == 1) {
                    cropOffset = offsetData[0];
                } else if (offsetSize > 1) {
                    cropOffset = offsetData[i - axis];
                }
            }
            offsets[i] = cropOffset;
        }
        std::vector<int> seperateInputDims;
        std::vector<int> seperateOutputDims;
        std::vector<int> seperateOffsets;
        std::vector<int> seperateOutputStrides;
        std::vector<int> seperateInputStrides;
        int remainStride[MNN_MAX_TENSOR_DIM];
        int remainStrideSize;
        computeOffsetRegion(TensorUtils::getDescribe(outputs[0]), input, outputs[0], input, offsets, rightOffset, seperateInputDims,
                            seperateOutputDims, seperateOffsets, seperateInputStrides, seperateOutputStrides,
                            remainStride, remainStrideSize);
        return true;
    }
};

class GeometryPad : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input            = inputs[0];
        auto output           = outputs[0];
        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->regions.clear();
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        for (int i=0; i<input->dimensions(); ++i) {
            if (input->length(i) == 0) {
                return true;
            }
        }
        auto paddings         = inputs[1];
        auto paddingPtr       = paddings->host<int32_t>();
        auto dimensions       = input->dimensions();
        std::vector<int> pads(dimensions);
        std::vector<int> padRights(dimensions);
        for (int i = 0; i < dimensions; ++i) {
            pads[i] = paddingPtr[2 * i];
            padRights[i] = paddingPtr[2 * i + 1];
        }
        auto param     = op->main_as_PadParam();
        std::vector<int> seperateInputDims;
        std::vector<int> seperateOutputDims;
        std::vector<int> seperateOffsets;
        std::vector<int> seperateOutputStrides;
        std::vector<int> seperateInputStrides;
        int remainStride[MNN_MAX_TENSOR_DIM];
        int remainStrideSize;

        computeOffsetRegion(outputDes, output, input, input, pads, padRights, seperateOutputDims, seperateInputDims,
                            seperateOffsets, seperateOutputStrides, seperateInputStrides, remainStride, remainStrideSize);
        int remainSize =
            OpCommonUtils::computeStride(remainStride, seperateOutputDims.data(), remainStrideSize);

        // Revert region
        for (auto& reg : outputDes->regions) {
            auto t  = reg.dst;
            reg.dst = reg.src;
            reg.src = t;
        }
        auto mode = PadValueMode_CONSTANT;
        if (param) {
            mode = param->mode();
        }
        auto padInput = input;
        int sStrideDiff = 1;
        if (PadValueMode_CONSTANT == mode) {
            if (inputs.size() <= 2) {
                return true;
            }
            // Check Zero for inputs[2]
            bool zero = false;
            auto type = inputs[2]->getType();
            if (!TensorUtils::getDescribe(inputs[2])->isMutable && inputs[2]->deviceId() == 0) {
                switch (type.code) {
                    case halide_type_int:
                    {
                        if (type.bits == 8) {
                            zero = inputs[2]->host<int8_t>()[0] == 0;
                        } else if (type.bits == 32) {
                            zero = inputs[2]->host<int32_t>()[0] == 0;
                        }
                    }
                        break;
                    case halide_type_uint:
                    {
                        if (type.bits == 8) {
                            zero = inputs[2]->host<uint8_t>()[0] == 0;
                        } else if (type.bits == 32) {
                            zero = inputs[2]->host<uint32_t>()[0] == 0;
                        }
                    }
                        break;
                    case halide_type_float:
                    {
                        zero = inputs[2]->host<float>()[0] == 0.0f;
                    }
                        break;
                    default:
                        break;
                }
            }
            if (zero) {
                return true;
            }
            padInput = inputs[2];
            sStrideDiff = 0;
        }
        // For Reflect and Mirror
        /* Ref: https://www.tensorflow.org/api_docs/python/tf/pad
        If mode is "REFLECT"
         then both paddings[D, 0] and paddings[D, 1] must be no greater than tensor.dim_size(D) - 1.
        If mode is "SYMMETRIC"
         then both paddings[D, 0] and paddings[D, 1] must be no greater than tensor.dim_size(D).*/
        int extraSub = 0;
        if (PadValueMode_REFLECT == mode) {
            extraSub = 1;
        }
        std::vector<int> rightPads(seperateOffsets.size());
        for (int i = 0; i < rightPads.size(); ++i) {
            rightPads[i] = seperateOutputDims[i] - seperateInputDims[i] - seperateOffsets[i];
        }
        std::vector<int> padRegion;
        for (int i = remainStrideSize; i < seperateInputStrides.size(); ++i) {
            // 0: center, 1: left, 2: right
            int r = 1;
            if (seperateOffsets[i] > 0) {
                r++;
            }
            if (rightPads[i] > 0) {
                r++;
            }
            padRegion.emplace_back(r);
        }
        int padRegionMod[MNN_MAX_TENSOR_DIM];
        int regionSize      = OpCommonUtils::computeStride(padRegionMod, padRegion.data(), padRegion.size());
        int remainDimOffset = (int)remainStrideSize;
        std::vector<int> padCord(padRegion.size());
        std::vector<int> cords(remainStrideSize);
        for (int pos = 0; pos < remainSize; ++pos) {
            int dstBasicOffset = 0;
            int srcBasicOffset = 0;
            OpCommonUtils::unravelIndexHelper(cords.data(), remainStride, remainDimOffset, pos);
            for (int i = 0; i < cords.size(); ++i) {
                // cords is the pos of output
                dstBasicOffset += cords[i] * seperateOutputStrides[i];
                // compute cords for input
                int inputPos = cords[i] - seperateOffsets[i];
                if (inputPos >= seperateInputDims[i]) {
                    // last -> last - extraSub - 1
                    inputPos = (seperateInputDims[i] - inputPos) + seperateInputDims[i] - extraSub - 1;
                }
                if (inputPos < 0) {
                    // -1 -> 0 + extraSub
                    inputPos = -inputPos + 1 + extraSub;
                }
                srcBasicOffset += inputPos * seperateInputStrides[i];
            }
            for (int index = 1; index < regionSize; ++index) {
                int dstOffset = dstBasicOffset;
                int srcOffset = srcBasicOffset;
                OpCommonUtils::unravelIndexHelper(padCord.data(), padRegionMod, padRegion.size(), index);
                Tensor::InsideDescribe::Region region;
                region.origin  = padInput;
                int sizeOffset = 3 - (int)padRegion.size();
                for (int i = 0; i < padRegion.size(); ++i) {
                    int di = sizeOffset + i;
                    int si = remainDimOffset + i;
                    switch (padCord[i]) {
                        case 0:
                            // center part: dst: start(offset) -> src: 0
                            dstOffset += seperateOffsets[si] * seperateOutputStrides[si];
                            region.size[di]       = seperateInputDims[si];
                            region.src.stride[di] = seperateInputStrides[si];
                            region.dst.stride[di] = seperateOutputStrides[si];
                            break;
                        case 2:
                            // right part: dst: start + inputDim -> src: inputDim - 1 - extra
                            dstOffset += (seperateOffsets[si] + seperateInputDims[si]) * seperateOutputStrides[si];
                            srcOffset += (seperateInputDims[si] - 1 - extraSub) * seperateInputStrides[si];
#define SET_SIZE(dst, size) \
if (mode == PadValueMode_REFLECT || mode == PadValueMode_SYMMETRIC) { \
if (size > seperateInputDims[si] - extraSub)  { \
MNN_ERROR("padding size is too large, result is undefined!\n(padding <= dim - 1) on REFLECT mode, (padding <= dim) on SYMMETRIC mode\n");  \
} \
dst = ALIMIN(size, seperateInputDims[si] - extraSub);\
} else { dst = size; }
                            SET_SIZE(region.size[di], rightPads[si])
                            region.src.stride[di] = -seperateInputStrides[si];
                            region.dst.stride[di] = seperateOutputStrides[si];
                            break;
                        case 1:
                            // offset = 0 means right part, offset > 0 means left part
                            if (seperateOffsets[si] > 0) {
                                // left part: dst: 0 -> src: seperateOffsets  + extra - 1
                                auto srcPos = seperateOffsets[si] - 1 + extraSub;
                                if (mode == PadValueMode_EDGE) {
                                    srcPos = 0;
                                }
                                srcOffset += srcPos * seperateInputStrides[si];
                                SET_SIZE(region.size[di], seperateOffsets[si])
                                region.src.stride[di] = -seperateInputStrides[si];
                                region.dst.stride[di] = seperateOutputStrides[si];
                            } else {
                                // right part: dst: start + inputDim -> src: inputDim - 1 - extra
                                dstOffset += (seperateOffsets[si] + seperateInputDims[si]) * seperateOutputStrides[si];
                                srcOffset += (seperateInputDims[si] - 1 - extraSub) * seperateInputStrides[si];
                                SET_SIZE(region.size[di], rightPads[si])
                                region.src.stride[di] = -seperateInputStrides[si];
                                region.dst.stride[di] = seperateOutputStrides[si];
                            }
                            break;
                        default:
                            break;
                    }
                    if (padCord[i] != 0 && mode == PadValueMode_EDGE) {
                        region.src.stride[di] = 0;
                    }
                }
                region.src.offset = srcOffset;
                region.dst.offset = dstOffset;
                if (sStrideDiff == 0) {
                    region.src.offset = 0;
                    region.src.stride[0] = 0;
                    region.src.stride[1] = 0;
                    region.src.stride[2] = 0;
                }
                outputDes->regions.emplace_back(std::move(region));
            }
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryCrop);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Crop});
    std::shared_ptr<GeometryComputer> comp2(new GeometryPad);
    GeometryComputer::registerGeometryComputer(comp2, {OpType_Padding});
}

REGISTER_GEOMETRY(GeometryCrop, _create);

} // namespace MNN
