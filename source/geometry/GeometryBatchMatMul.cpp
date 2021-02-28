//
//  GeometryBatchMatMul.cpp
//  MNN
//
//  Created by MNN on 2020/07/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"

namespace MNN {
class GeometryBatchMatMul : public GeometryComputer {
public:

    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs, Context& context, CommandBuffer& res) const override {
        bool transposeA = false;
        bool transposeB = false;
        
        auto input0          = inputs[0];
        auto input1          = inputs[1];
        auto output          = outputs[0];

        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->regions.clear();
        // Fill output by zero if one of inputs is empty.
        if (input0->elementSize() == 0 || input1->elementSize() == 0) {
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            return true;
        }
        if (outputs[0]->dimensions() == 2) {
            // Use normal MatMul
            Command cmd;
            cmd.op      = op;
            cmd.inputs  = std::move(inputs);
            cmd.outputs = std::move(outputs);
            res.command.emplace_back(std::move(cmd));
            return true;
        }
        // Broadcast matmul don't support bias
        MNN_ASSERT(inputs.size() == 2);
        // Split MatMul
        if (op->type() == OpType_BatchMatMul) {
            auto param = op->main_as_BatchMatMulParam();
            transposeA = param->adjX();
            transposeB = param->adjY();
        } else {
            auto param = op->main_as_MatMul();
            transposeA = param->transposeA();
            transposeB = param->transposeB();
        }
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        auto o0Dim = output->dimensions();
        int input0_end1 = input0->length(input0->dimensions()-2);
        int input0_end0 = input0->length(input0->dimensions()-1);
        int input1_end1 = input1->length(input1->dimensions()-2);
        int input1_end0 = input1->length(input1->dimensions()-1);
        // Compute BroastCast Dims
        auto dimOffset = o0Dim - 2;
        const int maxDimensions = dimOffset;
        std::vector<int> outputStrides(maxDimensions);
        std::vector<int> input0Strides(maxDimensions, 0);
        std::vector<int> input1Strides(maxDimensions, 0);
        auto i0Offset = output->dimensions() - input0->dimensions();
        auto i1Offset = output->dimensions() - input1->dimensions();
        int totalSize = 1;
        int i0Size = 1;
        int i1Size = 1;
        for (int i = maxDimensions - 1; i >=0 ; --i) {
            outputStrides[i] = totalSize;
            totalSize *= output->length(i);
            if (i >= i0Offset && input0->length(i - i0Offset) > 1) {
                input0Strides[i] = i0Size;
                i0Size *= input0->length(i - i0Offset);
            }
            if (i >= i1Offset && input1->length(i - i1Offset) > 1) {
                input1Strides[i] = i1Size;
                i1Size *= input1->length(i - i1Offset);
            }
        }
        std::vector<uint8_t> opBuffer;
        {
            flatbuffers::FlatBufferBuilder builder;
            MatMulBuilder builder_(builder);
            builder_.add_transposeA(transposeA);
            builder_.add_transposeB(transposeB);
            auto mainOffset = builder_.Finish().Union();
            OpBuilder opB(builder);
            opB.add_type(OpType_MatMul);
            opB.add_main(mainOffset);
            opB.add_main_type(OpParameter_MatMul);
            builder.Finish(opB.Finish());
            opBuffer.resize(builder.GetSize());
            ::memcpy(opBuffer.data(), builder.GetBufferPointer(), builder.GetSize());
        }
        for (int index = 0; index < totalSize; ++index) {
            // Unrool the cords
            auto c = index;
            i0Offset = 0;
            i1Offset = 0;
            for (int i=0; i<maxDimensions; ++i) {
                auto cord = c / outputStrides[i];
                i0Offset += input0Strides[i] * cord;
                i1Offset += input1Strides[i] * cord;
                c = c % outputStrides[i];
            }
            std::shared_ptr<Tensor> tmpInput0;
            {
                tmpInput0.reset(new Tensor);
                tmpInput0->buffer().type = halide_type_of<float>();
                tmpInput0->buffer().dimensions = 2;
                tmpInput0->setLength(0, input0_end1);
                tmpInput0->setLength(1, input0_end0);
                auto suboutputDes = TensorUtils::getDescribe(tmpInput0.get());
                suboutputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;

                Tensor::InsideDescribe::Region desReg;
                desReg.size[0] = 1;
                desReg.size[1] = input0_end1;
                desReg.size[2] = input0_end0;
                desReg.dst.offset = 0;
                desReg.dst.stride[0] = input0_end0*input0_end1;
                desReg.dst.stride[1] = input0_end0;
                desReg.dst.stride[2] = 1;
                desReg.src.offset = i0Offset*input0_end0*input0_end1;
                desReg.src.stride[0] = input0_end0*input0_end1;
                desReg.src.stride[1] = input0_end0;
                desReg.src.stride[2] = 1;
                desReg.origin = input0;
                suboutputDes->regions.emplace_back(std::move(desReg));
                
                res.extras.emplace_back(tmpInput0);
            }
            
            std::shared_ptr<Tensor> tmpInput1;
            {
                tmpInput1.reset(new Tensor);
                tmpInput1->buffer().type = halide_type_of<float>();
                tmpInput1->buffer().dimensions = 2;
                tmpInput1->setLength(0, input1_end1);
                tmpInput1->setLength(1, input1_end0);
                auto suboutputDes = TensorUtils::getDescribe(tmpInput1.get());
                suboutputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;

                Tensor::InsideDescribe::Region desReg;
                desReg.size[0] = 1;
                desReg.size[1] = input1_end1;
                desReg.size[2] = input1_end0;
                desReg.dst.offset = 0;
                desReg.dst.stride[0] = input1_end0*input1_end1;
                desReg.dst.stride[1] = input1_end0;
                desReg.dst.stride[2] = 1;
                desReg.src.offset = i1Offset*input1_end0*input1_end1;
                desReg.src.stride[0] = input1_end0*input1_end1;
                desReg.src.stride[1] = input1_end0;
                desReg.src.stride[2] = 1;
                desReg.origin = input1;
                suboutputDes->regions.emplace_back(std::move(desReg));
                
                res.extras.emplace_back(tmpInput1);
            }
            
            int dim0 = transposeA ? input0_end0 : input0_end1;
            int dim1 = transposeB ? input1_end1 : input1_end0;
            std::shared_ptr<Tensor> C;
            {
                // C = MatMul(B, A)
                C.reset(new Tensor);
                C->buffer().type = halide_type_of<float>();
                C->buffer().dimensions = 2;

                C->setLength(0, dim0);
                C->setLength(1, dim1);

                res.extras.emplace_back(C);
                Command cmd;
                cmd.buffer = opBuffer;
                cmd.inputs = {tmpInput0.get(), tmpInput1.get()};
                cmd.outputs = {C.get()};
                res.command.emplace_back(std::move(cmd));
            }
            
            {
                
                Tensor::InsideDescribe::Region desReg;
                desReg.size[0] = 1;
                desReg.size[1] = dim0;
                desReg.size[2] = dim1;
                desReg.dst.offset = index*dim0*dim1;
                desReg.dst.stride[0] = dim0*dim1;
                desReg.dst.stride[1] = dim1;
                desReg.dst.stride[2] = 1;
                desReg.src.offset = 0;
                desReg.src.stride[0] = dim0*dim1;
                desReg.src.stride[1] = dim1;
                desReg.src.stride[2] = 1;
                desReg.origin = C.get();
                outputDes->regions.emplace_back(std::move(desReg));
            }
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryBatchMatMul);
    GeometryComputer::registerGeometryComputer(comp, {OpType_MatMul});
}

REGISTER_GEOMETRY(GeometryBatchMatMul, _create);

} // namespace MNN
