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

namespace MNN {

class GeometryPooling3D : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto input = inputs[0];
        auto output = outputs[0];
        MNN_ASSERT(input->dimensions() == 5);
        auto kernelSize = op->main_as_Pool3D()->kernels();
        auto strideSize = op->main_as_Pool3D()->strides();
        auto padSize = op->main_as_Pool3D()->pads();
        auto poolType = op->main_as_Pool3D()->type();
        auto padType = op->main_as_Pool3D()->padType();

        const int kernelDepth = kernelSize->Get(0), kernelHeight = kernelSize->Get(1), kernelWidth = kernelSize->Get(2);
        const int strideDepth = strideSize->Get(0), strideHeight = strideSize->Get(1), strideWidth = strideSize->Get(2);
        const int padDepth = padSize->Get(0), padHeight = padSize->Get(1), padWidth = padSize->Get(2);
        const int outputDepth = output->length(2), outputHeight = output->length(3), outputWidth = output->length(4);
        const int inputDepth = input->length(2), inputHeight = input->length(3), inputWidth = input->length(4);
        const int channel = input->length(1), batch = input->length(0);
        std::shared_ptr<Tensor> reshapeInput;
        {
            reshapeInput.reset(Tensor::createDevice<float>({batch*inputDepth, channel, inputHeight, inputWidth}));
            auto outputDes = TensorUtils::getDescribe(reshapeInput.get());
            outputDes->regions.clear();
            outputDes->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            auto totalSlice = TensorUtils::makeFullSlice(input);
            outputDes->regions.emplace_back(std::move(totalSlice));
            res.extras.emplace_back(reshapeInput);
        }
        std::shared_ptr<Tensor> pool2dTmp1;
        {
            pool2dTmp1.reset(Tensor::createDevice<float>({batch*inputDepth, channel, outputHeight, outputWidth}));
            auto outputDes = TensorUtils::getDescribe(pool2dTmp1.get());
            outputDes->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(GeometryComputerUtils::makePool(builder, std::make_pair(kernelWidth, kernelHeight), std::make_pair(strideWidth, strideHeight), poolType, padType, std::make_pair(padWidth, padHeight), false));
            auto cmd = GeometryComputerUtils::makeCommand(builder, {reshapeInput.get()}, {pool2dTmp1.get()});
            res.extras.emplace_back(pool2dTmp1);
            res.command.emplace_back(std::move(cmd));
        }
        std::shared_ptr<Tensor> reshapeTmp1;
        {
            reshapeTmp1.reset(Tensor::createDevice<float>({batch, channel, inputDepth, outputHeight*outputWidth}));
            auto outputDes = TensorUtils::getDescribe(reshapeTmp1.get());
            outputDes->regions.clear();
            outputDes->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            auto totalSlice = TensorUtils::makeFullSlice(pool2dTmp1.get());
            outputDes->regions.emplace_back(std::move(totalSlice));
            res.extras.emplace_back(reshapeTmp1);
        }
        std::shared_ptr<Tensor> pool2dTmp2;
        {
            pool2dTmp2.reset(Tensor::createDevice<float>({batch, channel, outputDepth, outputHeight*outputWidth}));
            TensorUtils::getDescribe(pool2dTmp2.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            auto countType = AvgPoolCountType_DEFAULT;
            if (poolType == PoolType_AVEPOOL) {
                countType = AvgPoolCountType_EXCLUDE_PADDING;
            }
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(GeometryComputerUtils::makePool(builder, std::make_pair(1, kernelDepth), std::make_pair(1, strideDepth), poolType, padType, std::make_pair(0, padDepth), false, countType));
            auto cmd = GeometryComputerUtils::makeCommand(builder, {reshapeTmp1.get()}, {pool2dTmp2.get()});
            res.extras.emplace_back(pool2dTmp2);
            res.command.emplace_back(std::move(cmd));
        }
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
