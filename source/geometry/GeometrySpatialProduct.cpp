//
//  GeometrySpatialProduct.cpp
//  MNN
//
//  Created by MNN on 2020/07/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"

namespace MNN {
class GeometrySpatialProduct : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs, Context& context, CommandBuffer& res) const override {
        // Assume
        // bottom[0] dim CxHxW
        // bottom[1] dim 1xHxW
        // top[0]    dim CxHxW
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto input     = inputs[0];
        auto input1    = inputs[1];
        auto output    = outputs[0];
        
        int ib      = input->batch();
        int iw      = input->width();
        int ih      = input->height();
        int ic      = input->channel();
        
        auto ob = output->batch();
        auto oc = output->channel();
        auto oh = output->height();
        auto ow = output->width();
        auto inside = iw*ih;
        
        //input transform to NCHW format
        std::shared_ptr<Tensor> tmpInput;
        {
            tmpInput.reset(new Tensor);
            tmpInput->buffer().type = halide_type_of<float>();
            tmpInput->buffer().dimensions = 4;
            tmpInput->setLength(0, ib);
            tmpInput->setLength(1, ic);
            tmpInput->setLength(2, ih);
            tmpInput->setLength(3, iw);
            auto outputDes = TensorUtils::getDescribe(tmpInput.get());
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            outputDes->dimensionFormat = MNN_DATA_FORMAT_NCHW;

            Tensor::InsideDescribe::Region desReg;
            desReg.size[0] = ib;
            desReg.size[1] = ic;
            desReg.size[2] = inside;
            desReg.dst.offset = 0;
            desReg.dst.stride[0] = ic*inside;
            desReg.dst.stride[1] = inside;
            desReg.dst.stride[2] = 1;
            desReg.src.offset = 0;
            desReg.src.stride[0] = ic*inside;
            desReg.src.stride[1] = inside;
            desReg.src.stride[2] = 1;
            desReg.origin = input;
            outputDes->regions.emplace_back(std::move(desReg));
            
            res.extras.emplace_back(tmpInput);
        }
        
        //input1 broadcast to NCHW format
        std::shared_ptr<Tensor> tmpInput1;
        {
            tmpInput1.reset(new Tensor);
            tmpInput1->buffer().type = halide_type_of<float>();
            tmpInput1->buffer().dimensions = 4;
            tmpInput1->setLength(0, ib);
            tmpInput1->setLength(1, ic);
            tmpInput1->setLength(2, ih);
            tmpInput1->setLength(3, iw);
            auto outputDes = TensorUtils::getDescribe(tmpInput1.get());
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            outputDes->dimensionFormat = MNN_DATA_FORMAT_NCHW;

            Tensor::InsideDescribe::Region desReg;
            desReg.size[0] = ib;
            desReg.size[1] = ic;
            desReg.size[2] = inside;
            desReg.dst.offset = 0;
            desReg.dst.stride[0] = ic*inside;
            desReg.dst.stride[1] = inside;
            desReg.dst.stride[2] = 1;
            desReg.src.offset = 0;
            desReg.src.stride[0] = inside;
            desReg.src.stride[1] = 0;
            desReg.src.stride[2] = 1;
            desReg.origin = input1;
            outputDes->regions.emplace_back(std::move(desReg));
            
            res.extras.emplace_back(tmpInput1);
        }
        
        std::shared_ptr<Tensor> tmpOutput;
        {
            tmpOutput.reset(new Tensor);
            tmpOutput->buffer().type = halide_type_of<float>();
            tmpOutput->buffer().dimensions = 4;
            tmpOutput->setLength(0, ob);
            tmpOutput->setLength(1, oc);
            tmpOutput->setLength(2, oh);
            tmpOutput->setLength(3, ow);
            auto des = TensorUtils::getDescribe(tmpOutput.get());
            des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, tmpInput.get(), tmpInput1.get(), tmpOutput.get());
        
            res.extras.emplace_back(tmpOutput);
            res.command.emplace_back(std::move(cmd));
        }
        
        
        //transform to output
        {
            auto outputDes = TensorUtils::getDescribe(output);
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            Tensor::InsideDescribe::Region desReg;
            desReg.size[0] = ob;
            desReg.size[1] = oc;
            desReg.size[2] = inside;
            desReg.dst.offset = 0;
            desReg.dst.stride[0] = oc*inside;
            desReg.dst.stride[1] = inside;
            desReg.dst.stride[2] = 1;
            desReg.src.offset = 0;
            desReg.src.stride[0] = oc*inside;
            desReg.src.stride[1] = inside;
            desReg.src.stride[2] = 1;
            desReg.origin = tmpOutput.get();
            outputDes->regions.emplace_back(std::move(desReg));
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometrySpatialProduct);
    GeometryComputer::registerGeometryComputer(comp, {OpType_SpatialProduct});
}

REGISTER_GEOMETRY(GeometrySpatialProduct, _create);

} // namespace MNN
