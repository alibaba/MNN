//
//  NeuronAdapterRaster.cpp
//  MNN
//
//  Created by MNN on 2022/09/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NeuronAdapterRaster.hpp"
#include "core/OpCommonUtils.hpp"


namespace MNN {

ErrorCode NeuronAdapterRaster::buildReshape(const std::vector<Tensor *> &outputs) {
    mDatas.push_back(outputs[0]->shape());
    mNeuronAdapterBackend->dimsFormat<int>(mDatas.back(), TensorUtils::getDescribe(outputs[0])->dimensionFormat);
    const auto& regions = TensorUtils::getDescribe(outputs[0])->regions;
    std::vector<uint32_t> inputIdx(2);
    inputIdx[0] = mNeuronAdapterBackend->getTensorIdx(regions[0].origin);
    inputIdx[1] = buildVector(mDatas.back());
    return buildOperation(NEURON_RESHAPE, inputIdx, getTensorIdxs(outputs));
}

ErrorCode NeuronAdapterRaster::buildPermute(const std::vector<Tensor *> &outputs) {
    const auto input = TensorUtils::getDescribe(outputs[0])->regions[0].origin;
    std::vector<uint32_t> inputIdx(2);
    inputIdx[0] = mNeuronAdapterBackend->getTensorIdx(input);
    auto ishape = input->shape();
    auto oshape = outputs[0]->shape();
    // TODO
    mDatas.push_back(std::vector<int>(ishape.size()));
    inputIdx[1] = buildVector(mDatas.back());
    return buildOperation(NEURON_TRANSPOSE, inputIdx, getTensorIdxs(outputs));
}

ErrorCode NeuronAdapterRaster::buildTile(const std::vector<Tensor *> &outputs) {
    const auto input = TensorUtils::getDescribe(outputs[0])->regions[0].origin;
    auto ishape = input->shape();
    auto oshape = outputs[0]->shape();
    mDatas.push_back(std::vector<int>(ishape.size()));
    for (int i = 0; i < ishape.size(); i++) {
        mDatas.back()[i] = oshape[i] / ishape[i];
    }
    mNeuronAdapterBackend->dimsFormat(mDatas.back(), TensorUtils::getDescribe(input)->dimensionFormat);
    std::vector<uint32_t> inputIdx(2);
    inputIdx[0] = mNeuronAdapterBackend->getTensorIdx(input);
    inputIdx[1] = buildVector(mDatas.back());
    return buildOperation(NEURON_TILE, inputIdx, getTensorIdxs(outputs));
}

ErrorCode NeuronAdapterRaster::buildPad(const std::vector<Tensor *> &outputs) {
    const auto input = TensorUtils::getDescribe(outputs[0])->regions[0].origin;
    auto ishape = input->shape();
    auto oshape = outputs[0]->shape();
    mDatas.push_back(std::vector<int>(ishape.size() * 2));
    for (int i = 0; i < ishape.size(); i++) {
        mDatas.back()[2 * i] = 0;
        mDatas.back()[2 * i + 1] = oshape[i] - ishape[i];
    }
    if (!mNCHW && TensorUtils::getDescribe(input)->dimensionFormat != MNN_DATA_FORMAT_NHWC) {
        int padcr = mDatas.back()[3], padhr = mDatas.back()[5], padwr = mDatas.back()[7];
        mDatas.back()[3] = padhr;
        mDatas.back()[5] = padwr;
        mDatas.back()[7] = padcr;
    }
    std::vector<uint32_t> inputIdx(2);
    inputIdx[0] = mNeuronAdapterBackend->getTensorIdx(input);
    inputIdx[1] = buildConstant(mDatas.back().data(), mDatas.back().size() * sizeof(int),
                                NEURON_TENSOR_INT32, {static_cast<uint32_t>(ishape.size()), 2});
    return buildOperation(NEURON_PAD, inputIdx, getTensorIdxs(outputs));
}

ErrorCode NeuronAdapterRaster::buildSlice(const std::vector<Tensor *> &outputs) {
    const auto& region = TensorUtils::getDescribe(outputs[0])->regions[0];
    const auto input = region.origin;
    auto ishape = input->shape();
    auto oshape = outputs[0]->shape();
    int beginIdx = mDatas.size();
    // begin value
    mDatas.push_back(std::vector<int>(ishape.size()));
    int offset = region.src.offset;
    for (int i = ishape.size() - 1; i >= 0; i--) {
        mDatas.back()[i] = offset % ishape[i];
        offset /= ishape[i];
    }
    mDatas.push_back(std::vector<int>(ishape.size()));
    for (int i = 0; i < ishape.size(); i++) {
        mDatas.back()[i] = oshape[i];
    }
    mNeuronAdapterBackend->dimsFormat(mDatas[beginIdx], TensorUtils::getDescribe(input)->dimensionFormat);
    mNeuronAdapterBackend->dimsFormat(mDatas.back(), TensorUtils::getDescribe(input)->dimensionFormat);
    std::vector<uint32_t> inputIdx(3);
    inputIdx[0] = mNeuronAdapterBackend->getTensorIdx(input);
    inputIdx[1] = buildVector(mDatas[beginIdx]);
    inputIdx[2] = buildVector(mDatas.back());
    return buildOperation(NEURON_SLICE, inputIdx, getTensorIdxs(outputs));
}

ErrorCode NeuronAdapterRaster::buildDepthToSpace(const std::vector<Tensor *> &outputs) {
    const auto input = TensorUtils::getDescribe(outputs[0])->regions[0].origin;
    std::vector<uint32_t> inputIdx(3);
    inputIdx[0] = mNeuronAdapterBackend->getTensorIdx(input);
    int blockSize = outputs[0]->height() / input->height();
    inputIdx[1] = buildScalar(blockSize);
    inputIdx[2] = buildScalar(mNCHW);
    return buildOperation(NEURON_DEPTH_TO_SPACE, inputIdx, getTensorIdxs(outputs));
}

ErrorCode NeuronAdapterRaster::buildConcat(const std::vector<Tensor *> &outputs, int axis) {
    const auto& regions = TensorUtils::getDescribe(outputs[0])->regions;
    std::vector<uint32_t> inputIdx(regions.size()+1);
    for (int i = 0; i < regions.size(); i++) {
        inputIdx[i] = mNeuronAdapterBackend->getTensorIdx(regions[i].origin);
    }
    inputIdx[regions.size()] = buildScalar(formatAxis(axis, outputs[0]));
    return buildOperation(NEURON_CONCATENATION, inputIdx, getTensorIdxs(outputs));
}

NeuronAdapterRaster::NeuronAdapterRaster(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NeuronAdapterCommonExecution(b, op) {
}
static void dumpRegion(const Tensor::InsideDescribe::Region& reg) {
    printf("\n{\nsize: [%d, %d, %d], origin: %p\n", reg.size[0], reg.size[1], reg.size[2], reg.origin);
    printf("src: { stride: [%d, %d, %d], offset: %d }\n", reg.src.stride[0],reg.src.stride[1],reg.src.stride[2],reg.src.offset);
    printf("dst: { stride: [%d, %d, %d], offset: %d }\n}\n", reg.dst.stride[0],reg.dst.stride[1],reg.dst.stride[2],reg.dst.offset);
}
ErrorCode NeuronAdapterRaster::onResize(const std::vector<Tensor *>& ____inputs, const std::vector<Tensor *> &outputs) {
    OpCommonUtils::rasterInputReset(____inputs, outputs[0]);
    const auto& regions = TensorUtils::getDescribe(outputs[0])->regions;
    if (regions.empty()) {
        return INVALID_VALUE;
    }
    const auto region = regions[0];
    const auto output = outputs[0];
#if 1
    printf("region.size = %d\n", regions.size());
    dumpRegion(regions[0]);
    regions[0].origin->printShape();
    outputs[0]->printShape();
#endif
    // propgate quant type to output
    //if (TensorUtils::getDescribe(region.origin)->quantAttr.get() &&
    //    TensorUtils::getDescribe(outputs[0])->usage == Tensor::InsideDescribe::Usage::NORMAL) {
    //    outputs[0]->buffer().type = region.origin->getType();
    //}
    // region_size = 1: reshape, transpose
    if (regions.size() == 1) {
        int inputSize = 1, outputSize = 1;
        for (int i = 0; i < region.origin->dimensions(); i++) {
            inputSize *= region.origin->length(i);
        }
        for (int i = 0; i < outputs[0]->dimensions(); i++) {
            outputSize *= outputs[0]->length(i);
        }
        // reshape, permute
        if (inputSize == outputSize) {
            // reshape
            if (TensorUtils::isCopyRegion(region)) {
                return buildReshape(outputs);
            }
            // transpose
            if (TensorUtils::isTransposeRegion(region)) {
                if (TensorUtils::getDescribe(region.origin)->dimensionFormat !=
                    TensorUtils::getDescribe(output)->dimensionFormat) {
                    // NeuronAdapter use same format, don't need convert tensor
                    return buildReshape(outputs);
                }
                return buildPermute(outputs);
            }
        }
        // tile, broadcast
        if (inputSize < outputSize) {
            if (TensorUtils::isTileRegion(region)) {
                //// TODO: find the way to judge the case
                //if (region.origin->channel() < output->channel()) {
                //    // tile for bianry input can skip, because nnapi support bianry broadcast
                //    return mNeuronAdapterBackend->replaceTensorWith(output, region.origin);
                //}
                return buildTile(outputs);
            }
            return buildPad(outputs);
        }
        // slice
        if (inputSize > outputSize) {
            // TODO: support strided_slice
            return buildSlice(outputs);
        }
        MNN_ERROR("[NeuronAdapter] Don't support Raster Mode.\n");
        return NOT_SUPPORT;
    }
    if (TensorUtils::isDepthToSpaceRegions(outputs[0])) {
        return buildDepthToSpace(outputs);
    }
    // region_size > 1: concat
    {
        int dim = output->dimensions();
        if (region.origin->dimensions() != dim) {
            return NOT_SUPPORT;
        }
        int axis = -1;
        for (int i = 0; i < output->dimensions(); i++) {
            if (region.origin->length(i) != output->length(i)) {
                if (axis >= 0) {
                    return NOT_SUPPORT;
                }
                axis = i;
            }
        }
        return buildConcat(outputs, axis);
    }
}

REGISTER_NeuronAdapter_OP_CREATOR(NeuronAdapterRaster, OpType_Raster)
} // namespace MNN
