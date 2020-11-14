//
//  GeometrySoftmax.cpp
//  MNN
//
//  Created by MNN on 2020/06/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"

namespace MNN {
class GeometrySoftmax : public GeometryComputer {
public:
    virtual std::vector<bool> onGetOutputVirtual(const Op* op, const std::vector<Tensor*>& inputs,
                                                 const std::vector<Tensor*>& outputs) const override {
        auto  axis = op->main_as_Axis()->axis();
        if (axis < 0) {
            axis = inputs[0]->dimensions() + axis;
        }
        
        if (axis == 1) {
            return std::vector<bool>(outputs.size(), false);
        }
        return std::vector<bool>(outputs.size(), true);
    }
    
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs, Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto input     = inputs[0];
        auto output    = outputs[0];
        auto dims      = input->buffer().dimensions;
        
        auto  axis = op->main_as_Axis()->axis();
        if (axis < 0) {
            axis = inputs[0]->dimensions() + axis;
        }
        
        if (axis == 1) {
            Command cmd;
            cmd.op      = op;
            cmd.inputs  = std::move(inputs);
            cmd.outputs = std::move(outputs);
            res.command.emplace_back(std::move(cmd));
            return true;
        }
        
        int inside  = 1;
        int outside = 1;
        int channel = 1;
        for (int i = 0; i < axis; ++i) {
            outside *= input->length(i);
        }
        channel = input->length(axis);
        for (int i = axis + 1; i < dims; ++i) {
            inside *= input->length(i);
        }

        //input transform to NCHW format
        std::shared_ptr<Tensor> tmpInput;
        {
            tmpInput.reset(Tensor::createDevice<float>({outside, channel, inside}));
            auto outputDes = TensorUtils::getDescribe(tmpInput.get());
            outputDes->regions.clear();
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
            desReg.origin = input;
            outputDes->regions.emplace_back(std::move(desReg));
            
            res.extras.emplace_back(tmpInput);
        }
        
        //reduction max, axis=1
        std::shared_ptr<Tensor> maxValue;
        {
            maxValue.reset(Tensor::createDevice<float>({outside, 1, inside}));
            res.extras.emplace_back(maxValue);
            res.command.emplace_back(GeometryComputerUtils::makeReduce(ReductionType_MAXIMUM, tmpInput.get(), maxValue.get()));
        }
        
        //broadcast reduction axis dim
        std::shared_ptr<Tensor> maxBroadValue;
        {
            maxBroadValue.reset(Tensor::createDevice<float>({outside, channel, inside}));
            auto outputDes = TensorUtils::getDescribe(maxBroadValue.get());
            outputDes->regions.clear();
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
            desReg.src.stride[0] = inside;
            desReg.src.stride[1] = 0;
            desReg.src.stride[2] = 1;
            desReg.origin = maxValue.get();
            outputDes->regions.emplace_back(std::move(desReg));
            
            res.extras.emplace_back(maxBroadValue);
        }

        //sub
        std::shared_ptr<Tensor> subMaxValue;
        {
            subMaxValue.reset(Tensor::createDevice<float>({outside, channel, inside}));
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_SUB, tmpInput.get(), maxBroadValue.get(), subMaxValue.get());
            res.extras.emplace_back(subMaxValue);
            res.command.emplace_back(std::move(cmd));
        }
        //exp
        std::shared_ptr<Tensor> expValue;
        {
            expValue.reset(Tensor::createDevice<float>({outside, channel, inside}));
            auto cmd = GeometryComputerUtils::makeUnary(UnaryOpOperation_EXP, subMaxValue.get(), expValue.get());
            res.extras.emplace_back(expValue);
            res.command.emplace_back(std::move(cmd));
            
        }
        
        //reduction sum, axis=2, only support NCHW
        std::shared_ptr<Tensor> sumValue;
        {
            sumValue.reset(Tensor::createDevice<float>({outside, 1, inside}));
            res.extras.emplace_back(sumValue);
            res.command.emplace_back(GeometryComputerUtils::makeReduce(ReductionType_SUM, expValue.get(), sumValue.get()));
        }
        
        //broadcast reduction axis dim
        std::shared_ptr<Tensor> sumBroadValue;
        {
            sumBroadValue.reset(Tensor::createDevice<float>({outside, channel, inside}));
            auto outputDes = TensorUtils::getDescribe(sumBroadValue.get());
            outputDes->regions.clear();
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
            desReg.src.stride[0] = inside;
            desReg.src.stride[1] = 0;
            desReg.src.stride[2] = 1;
            desReg.origin = sumValue.get();
            outputDes->regions.emplace_back(std::move(desReg));

            res.extras.emplace_back(sumBroadValue);
        }

        //div
        std::shared_ptr<Tensor> tmpOutput;
        {
            tmpOutput.reset(Tensor::createDevice<float>({outside, channel, inside}));
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_REALDIV, expValue.get(), sumBroadValue.get(), tmpOutput.get());
            res.extras.emplace_back(tmpOutput);
            res.command.emplace_back(std::move(cmd));
        }

        //transform to output
        {
            auto outputDes = TensorUtils::getDescribe(output);
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
            desReg.origin = tmpOutput.get();
            outputDes->regions.emplace_back(std::move(desReg));
        }
        return true;
    }
};

static void _create() {
//    std::shared_ptr<GeometryComputer> comp(new GeometrySoftmax);
//    GeometryComputer::registerGeometryComputer(comp, {OpType_Softmax});
}

REGISTER_GEOMETRY(GeometrySoftmax, _create);

} // namespace MNN
