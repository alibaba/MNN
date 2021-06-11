//
//  GeometryCosineSimilarity.cpp
//  MNN
//
//  Created by MNN on 2020/07/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"

namespace MNN {
class GeometryCosineSimilarity : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs, Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(3 <= inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto input0          = inputs[0];
        auto input1          = inputs[1];
        auto dimTensor = inputs[2];
        const auto dim = dimTensor->host<int32_t>()[0];
        MNN_ASSERT(dim == 1);
        auto output          = outputs[0];
        
        int dimensions = input0->dimensions();
        int outside = 1;
        int channel = 1;
        int inside = 1;
        for(int i=0; i<dim; i++) {
            outside *= input0->length(i);
        }
        channel = input0->length(dim);
        for(int i=dim+1; i<dimensions; i++) {
            inside *= input0->length(i);
        }
        auto dimType = input0->getDimensionType();
        
            
        //input0 transform to NCHW format
        std::shared_ptr<Tensor> tmpInput0;
        {
            tmpInput0.reset(Tensor::createDevice<float>({outside, channel, inside}, dimType));
            auto outputDes = TensorUtils::getDescribe(tmpInput0.get());
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;

            Tensor::InsideDescribe::Region desReg;
            desReg.size[0] = outside;
            desReg.size[1] = channel;
            desReg.size[2] = inside;
            desReg.dst.offset = 0;
            desReg.dst.stride[0] = channel*inside;
            desReg.dst.stride[1] = inside;
            desReg.dst.stride[2] = 1;
            desReg.src.offset = 0;
            desReg.src.stride[0] = channel*inside;
            desReg.src.stride[1] = inside;
            desReg.src.stride[2] = 1;
            desReg.origin = input0;
            outputDes->regions.emplace_back(std::move(desReg));

            res.extras.emplace_back(tmpInput0);
        }
        
        //input1 transform to NCHW format
        std::shared_ptr<Tensor> tmpInput1;
        {
            tmpInput1.reset(Tensor::createDevice<float>({outside, channel, inside}, dimType));
            auto outputDes = TensorUtils::getDescribe(tmpInput1.get());
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            outputDes->dimensionFormat = MNN_DATA_FORMAT_NCHW;

            Tensor::InsideDescribe::Region desReg;
            desReg.size[0] = outside;
            desReg.size[1] = channel;
            desReg.size[2] = inside;
            desReg.dst.offset = 0;
            desReg.dst.stride[0] = channel*inside;
            desReg.dst.stride[1] = inside;
            desReg.dst.stride[2] = 1;
            desReg.src.offset = 0;
            desReg.src.stride[0] = channel*inside;
            desReg.src.stride[1] = inside;
            desReg.src.stride[2] = 1;
            desReg.origin = input1;
            outputDes->regions.emplace_back(std::move(desReg));
            
            res.extras.emplace_back(tmpInput1);
        }
        
        //input0*input0
        std::shared_ptr<Tensor> tmpInput0x0;
        {
            tmpInput0x0.reset(Tensor::createDevice<float>({outside, channel, inside}, dimType));
            auto des = TensorUtils::getDescribe(tmpInput0x0.get());
            des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, tmpInput0.get(), tmpInput0.get(), tmpInput0x0.get());
            
            res.extras.emplace_back(tmpInput0x0);
            res.command.emplace_back(std::move(cmd));
        }
        
        //input0*input1
        std::shared_ptr<Tensor> tmpInput0x1;
        {
            tmpInput0x1.reset(Tensor::createDevice<float>({outside, channel, inside}, dimType));
            auto des = TensorUtils::getDescribe(tmpInput0x1.get());
            des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, tmpInput0.get(), tmpInput1.get(), tmpInput0x1.get());
            
            res.extras.emplace_back(tmpInput0x1);
            res.command.emplace_back(std::move(cmd));
        }
        
        //input1*input1
        std::shared_ptr<Tensor> tmpInput1x1;
        {
            tmpInput1x1.reset(Tensor::createDevice<float>({outside, channel, inside}, dimType));
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, tmpInput1.get(), tmpInput1.get(), tmpInput1x1.get());
            
            res.extras.emplace_back(tmpInput1x1);
            res.command.emplace_back(std::move(cmd));
        }

        //reduction sum, axis=1, only support NCHW
        std::shared_ptr<Tensor> sumValue0x0;
        {
            sumValue0x0.reset(Tensor::createDevice<float>({outside, 1, inside}, dimType));
            auto des = TensorUtils::getDescribe(sumValue0x0.get());
            auto cmd = GeometryComputerUtils::makeReduce(ReductionType_SUM, tmpInput0x0.get(), sumValue0x0.get());
            res.extras.emplace_back(sumValue0x0);
            res.command.emplace_back(std::move(cmd));
        }

        //reduction sum, axis=1, only support NCHW
        std::shared_ptr<Tensor> sumValue0x1;
        {
            sumValue0x1.reset(Tensor::createDevice<float>({outside, 1, inside}, dimType));
            auto des = TensorUtils::getDescribe(sumValue0x1.get());
            auto cmd = GeometryComputerUtils::makeReduce(ReductionType_SUM, tmpInput0x1.get(), sumValue0x1.get());
            res.extras.emplace_back(sumValue0x1);
            res.command.emplace_back(std::move(cmd));
        }
        
        //reduction sum, axis=1, only support NCHW
        std::shared_ptr<Tensor> sumValue1x1;
        {
            sumValue1x1.reset(Tensor::createDevice<float>({outside, 1, inside}, dimType));
            auto des = TensorUtils::getDescribe(sumValue1x1.get());
            
            auto cmd = GeometryComputerUtils::makeReduce(ReductionType_SUM, tmpInput1x1.get(), sumValue1x1.get());
        
            res.extras.emplace_back(sumValue1x1);
            res.command.emplace_back(std::move(cmd));
        }
        
        //sumValue0x0 * sumValue1x1
        std::shared_ptr<Tensor> mulValue0x0_1x1;
        {
            mulValue0x0_1x1.reset(Tensor::createDevice<float>({outside, 1, inside}, dimType));
            auto des = TensorUtils::getDescribe(mulValue0x0_1x1.get());
            
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, sumValue0x0.get(), sumValue1x1.get(), mulValue0x0_1x1.get());
            
            res.extras.emplace_back(mulValue0x0_1x1);
            res.command.emplace_back(std::move(cmd));
        }
        
        //add eps
        std::shared_ptr<Tensor> mulValue0x0_1x1_eps;
        {
            mulValue0x0_1x1_eps.reset(Tensor::createDevice<float>({outside, 1, inside}, dimType));
            auto des = TensorUtils::getDescribe(mulValue0x0_1x1_eps.get());
            
            const float eps         = 1e-8f;
            auto epsTensor = context.allocConst(op, {1}, halide_type_of<float>());
            epsTensor.get()->host<float>()[0] = eps;
            
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, mulValue0x0_1x1.get(), epsTensor.get(), mulValue0x0_1x1_eps.get());
            
            res.extras.emplace_back(mulValue0x0_1x1_eps);
            res.command.emplace_back(std::move(cmd));
        }

        //sqrt(sumValue0x0 * sumValue1x1 + eps)
        std::shared_ptr<Tensor> sqrtMulValue;
        {
            sqrtMulValue.reset(Tensor::createDevice<float>({outside, 1, inside}, dimType));
            auto des = TensorUtils::getDescribe(sqrtMulValue.get());
            
            auto cmd = GeometryComputerUtils::makeUnary(UnaryOpOperation_SQRT, mulValue0x0_1x1_eps.get(), sqrtMulValue.get());
            
            res.extras.emplace_back(sqrtMulValue);
            res.command.emplace_back(std::move(cmd));
        }
        //div
        std::shared_ptr<Tensor> tmpOutput;
        {
            tmpOutput.reset(Tensor::createDevice<float>({outside, 1, inside}, dimType));
            auto des = TensorUtils::getDescribe(tmpOutput.get());
            
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_REALDIV, sumValue0x1.get(), sqrtMulValue.get(), tmpOutput.get());
            
            res.extras.emplace_back(tmpOutput);
            res.command.emplace_back(std::move(cmd));
        }
        //transform to output
        {
            auto outputDes = TensorUtils::getDescribe(output);
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            Tensor::InsideDescribe::Region desReg;
            desReg.size[0] = 1;
            desReg.size[1] = outside;
            desReg.size[2] = inside;
            desReg.dst.offset = 0;
            desReg.dst.stride[0] = outside*inside;
            desReg.dst.stride[1] = inside;
            desReg.dst.stride[2] = 1;
            desReg.src.offset = 0;
            desReg.src.stride[0] = outside*inside;
            desReg.src.stride[1] = inside;
            desReg.src.stride[2] = 1;
            desReg.origin = tmpOutput.get();
            outputDes->regions.emplace_back(std::move(desReg));
        }
        return true;
        
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryCosineSimilarity);
    GeometryComputer::registerGeometryComputer(comp, {OpType_CosineSimilarity});
}

REGISTER_GEOMETRY(GeometryCosineSimilarity, _create);

} // namespace MNN
