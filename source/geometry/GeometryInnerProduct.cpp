//
//  GeometryInnerProduct.cpp
//  MNN
//
//  Created by MNN on 2020/05/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/ConvolutionCommon.hpp"
#include "ConvertUtils.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
namespace MNN {
class GeometryInnerProduct : public GeometryComputer {
public:

    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs, Context& context, CommandBuffer& res) const override {
        auto parameter  = op->main_as_InnerProduct();
        int outputCount = parameter->outputCount();
        int srcCount    = parameter->weight()->size() / outputCount;

        MNN_ASSERT(inputs.size() == 1);
        MNN_ASSERT(outputs.size() == 1);
        auto input = inputs[0];
        auto output = outputs[0];
        int inputDims = input->dimensions();
        int outputDims = output->dimensions();
        MNN_ASSERT(inputDims >= 2);
        MNN_ASSERT(outputDims == 2);
        MNN_ASSERT(output->length(1) == outputCount);
        
        int batch = output->length(0);
        MNN_ASSERT(input->length(0) == batch);
        int mulNum = 1;
        for(int i=1; i < inputDims; i++) {
            mulNum *= input->length(i);
        }
        if (srcCount != mulNum) {
            return false;
        }

        Tensor* A = nullptr;
        Tensor* B = nullptr;
        {
            std::shared_ptr<Tensor> tmpInput(new Tensor);
            tmpInput->buffer().type = halide_type_of<float>();
            tmpInput->buffer().dimensions = 2;
            tmpInput->setLength(0, batch);
            tmpInput->setLength(1, srcCount);
            auto des = TensorUtils::getDescribe(tmpInput.get());
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            des->regions.clear();
            des->regions.reserve(1);

            Tensor::InsideDescribe::Region region;
            region.origin = input;
            region.size[0] = 1;
            region.size[1] = batch;
            region.size[2] = srcCount;
            region.src.offset = 0;
            region.dst.offset = 0;
            region.src.stride[0] = 1;
            region.dst.stride[0] = 1;
            region.src.stride[1] = srcCount;
            region.dst.stride[1] = srcCount;
            region.src.stride[2] = 1;
            region.dst.stride[2] = 1;
            des->regions.emplace_back(std::move(region));

            A = tmpInput.get();
            res.extras.emplace_back(tmpInput);
        }
        
        std::shared_ptr<Tensor> tmpOutput(new Tensor);
        std::shared_ptr<Tensor> C(new Tensor);
        auto constTensors = context.searchConst(op);
        Tensor* weight = nullptr;
        Tensor* bias = nullptr;
        if (!constTensors.empty()) {
            MNN_ASSERT(constTensors.size() == 2);
            weight = constTensors[0].get();
            bias = constTensors[1].get();
        } else {
            auto weightTensor = context.allocConst(op, {outputCount, srcCount}, halide_type_of<float>());
            ::memcpy(weightTensor.get()->host<float>(), parameter->weight()->data(), parameter->weight()->size()*sizeof(float));
            weight = weightTensor.get();
            auto biasTensor = context.allocConst(op, {batch, outputCount}, halide_type_of<float>());
            ::memcpy(biasTensor.get()->host<float>(), parameter->bias()->data(), parameter->bias()->size()*sizeof(float));
            bias = biasTensor.get();
        }
        {
            B = weight;

            C->buffer().type = halide_type_of<float>();
            C->buffer().dimensions = 2;
            C->setLength(0, batch);
            C->setLength(1, outputCount);

            auto cmd = GeometryComputerUtils::makeMatMul(A, B, C.get(), nullptr, false, true);
            res.extras.emplace_back(C);
            res.command.emplace_back(std::move(cmd));
        }

        {
            tmpOutput->buffer().type = halide_type_of<float>();
            tmpOutput->buffer().dimensions = 2;
            tmpOutput->setLength(0, batch);
            tmpOutput->setLength(1, outputCount);
            
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, C.get(), bias, tmpOutput.get());
            res.extras.emplace_back(tmpOutput);
            res.command.emplace_back(std::move(cmd));
        }
        
        {
            auto des = TensorUtils::getDescribe(output);
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->regions.clear();
            des->regions.reserve(1);

            Tensor::InsideDescribe::Region region;
            region.origin = tmpOutput.get();
            region.size[0] = 1;
            region.size[1] = batch;
            region.size[2] = outputCount;
            region.src.offset = 0;
            region.dst.offset = 0;
            region.src.stride[0] = 1;
            region.dst.stride[0] = 1;
            region.src.stride[1] = outputCount;
            region.dst.stride[1] = outputCount;
            region.src.stride[2] = 1;
            region.dst.stride[2] = 1;
            des->regions.emplace_back(std::move(region));
        }
        
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryInnerProduct);
    GeometryComputer::registerGeometryComputer(comp, {OpType_InnerProduct});
}

REGISTER_GEOMETRY(GeometryInnerProduct, _create);

} // namespace MNN
