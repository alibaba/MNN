//
//  GeometryPooling3D.cpp
//  MNN
//
//  Created by MNN on 2020/7/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "ConvertUtils.hpp"
#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/Macro.h"

namespace MNN {

class GeometryPooling3D : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto input = inputs[0];
        auto output = outputs[0];
        auto isGlobal = op->main_as_Pool3D()->isGlobal();
        auto kernelSize = op->main_as_Pool3D()->kernels();
        auto strideSize = op->main_as_Pool3D()->strides();
        auto padSize = op->main_as_Pool3D()->pads();
        auto poolType = op->main_as_Pool3D()->type();
        auto padType = op->main_as_Pool3D()->padType();

        const int inputDepth = input->length(2), inputHeight = input->length(3), inputWidth = input->length(4);
        const int outputDepth = output->length(2), outputHeight = output->length(3), outputWidth = output->length(4);
        const int channel = input->length(1), batch = input->length(0);
        const int inputArea = inputHeight * inputWidth, outputArea = outputHeight * outputWidth;
        int kernelDepth = 0, kernelHeight = 0, kernelWidth = 0,
            strideDepth = 0, strideHeight = 0, strideWidth = 0,
            padDepth = 0, padHeight = 0, padWidth = 0;
        if (isGlobal) {
            // 2D GlobalPool
            if (inputs[0]->dimensions() < 5) {
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(GeometryComputerUtils::makePool(builder, std::make_pair(kernelWidth, kernelHeight), std::make_pair(strideWidth, strideHeight), poolType, padType, std::make_pair(padWidth, padHeight), isGlobal));
                auto cmd = GeometryComputerUtils::makeCommand(builder, {inputs[0]}, {outputs[0]});
                res.command.emplace_back(std::move(cmd));
                return true;
            }
        } else {
            kernelDepth = kernelSize->Get(0), kernelHeight = kernelSize->Get(1), kernelWidth = kernelSize->Get(2);
            strideDepth = strideSize->Get(0), strideHeight = strideSize->Get(1), strideWidth = strideSize->Get(2);
            padDepth = padSize->Get(0), padHeight = padSize->Get(1), padWidth = padSize->Get(2);
        }
        // [N C ID IH IW] -> [N ID C IH IW]
        std::shared_ptr<Tensor> transposeInput;
        {
            transposeInput.reset(Tensor::createDevice<float>({batch*inputDepth, channel, inputHeight, inputWidth}));
            auto outputDes = TensorUtils::getDescribe(transposeInput.get());
            outputDes->regions.clear();
            outputDes->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            for (int i = 0; i < batch; i++) {
                int offset = i * channel * inputDepth * inputArea;
                Tensor::InsideDescribe::Region region;
                region.origin = input;
                region.size[0] = inputDepth;
                region.size[1] = channel;
                region.size[2] = inputArea;
                region.src.offset = offset;
                region.src.stride[0] = inputArea;
                region.src.stride[1] = inputArea * inputDepth;
                region.src.stride[2] = 1;
                region.dst.offset = offset;
                region.dst.stride[0] = inputArea * channel;
                region.dst.stride[1] = inputArea;
                region.dst.stride[2] = 1;
                outputDes->regions.emplace_back(std::move(region));
            }
            res.extras.emplace_back(transposeInput);
        }
        // pool hw: [N ID C IH IW] -> [N ID C OH OW]
        std::shared_ptr<Tensor> pool2dTmp1;
        {
            pool2dTmp1.reset(Tensor::createDevice<float>({batch*inputDepth, channel, outputHeight, outputWidth}));
            auto outputDes = TensorUtils::getDescribe(pool2dTmp1.get());
            outputDes->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(GeometryComputerUtils::makePool(builder, std::make_pair(kernelWidth, kernelHeight), std::make_pair(strideWidth, strideHeight), poolType, padType, std::make_pair(padWidth, padHeight), isGlobal));
            auto cmd = GeometryComputerUtils::makeCommand(builder, {transposeInput.get()}, {pool2dTmp1.get()});
            res.extras.emplace_back(pool2dTmp1);
            res.command.emplace_back(std::move(cmd));
        }
        // transpose: [N ID C OH OW] -> [N C ID OH*OW]
        std::shared_ptr<Tensor> transposeTmp1;
        {
            transposeTmp1.reset(Tensor::createDevice<float>({batch, channel, inputDepth, outputHeight*outputWidth}));
            auto outputDes = TensorUtils::getDescribe(transposeTmp1.get());
            outputDes->regions.clear();
            outputDes->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            for (int i = 0; i < batch; i++) {
                int offset = i * channel * inputDepth * outputArea;
                Tensor::InsideDescribe::Region region;
                region.origin = pool2dTmp1.get();
                region.size[0] = channel;
                region.size[1] = inputDepth;
                region.size[2] = outputArea;
                region.src.offset = 0;
                region.src.stride[0] = outputArea;
                region.src.stride[1] = outputArea * channel;
                region.src.stride[2] = 1;
                region.dst.offset = 0;
                region.dst.stride[0] = outputArea * inputDepth;
                region.dst.stride[1] = outputArea;
                region.dst.stride[2] = 1;
                outputDes->regions.emplace_back(std::move(region));
            }
            res.extras.emplace_back(transposeTmp1);
        }
        // pool depth: [N C ID OH*OW] -> [N C OD OH*OW]
        std::shared_ptr<Tensor> pool2dTmp2;
        {
            pool2dTmp2.reset(Tensor::createDevice<float>({batch, channel, outputDepth, outputHeight*outputWidth}));
            TensorUtils::getDescribe(pool2dTmp2.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            auto countType = AvgPoolCountType_DEFAULT;
            if (poolType == PoolType_AVEPOOL) {
                countType = AvgPoolCountType_EXCLUDE_PADDING;
            }
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(GeometryComputerUtils::makePool(builder, std::make_pair(1, kernelDepth), std::make_pair(1, strideDepth), poolType, padType, std::make_pair(0, padDepth), isGlobal, countType));
            auto cmd = GeometryComputerUtils::makeCommand(builder, {transposeTmp1.get()}, {pool2dTmp2.get()});
            res.extras.emplace_back(pool2dTmp2);
            res.command.emplace_back(std::move(cmd));
        }
        // reshape: [N C OD OH*OW] -> [N C OD OH OW]
        {
            auto outputDes = TensorUtils::getDescribe(output);
            outputDes->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            auto totalSlice = TensorUtils::makeFullSlice(pool2dTmp2.get());
            outputDes->regions.emplace_back(std::move(totalSlice));
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryPooling3D);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Pooling3D});
}

REGISTER_GEOMETRY(GeometryPooling3D, _create);

} // namespace MNN
