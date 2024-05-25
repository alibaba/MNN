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
    virtual bool onRecompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                             Context& context, CommandBuffer& cmd) const override {
        if (cmd.command.empty()) {
            return false;
        }
        if (cmd.command[0]->inputs.size() > 3) {
            // TODO: Support broadcast case
            return false;
        }
        bool transposeA = false;
        bool transposeB = false;
        
        auto input0          = inputs[0];
        auto input1          = inputs[1];
        Tensor* bias         = nullptr;
        auto output          = outputs[0];
        if (inputs.size() > 2) {
            bias = inputs[2];
        }
        if (input0->dimensions() < 2 || input1->dimensions() < 2) {
            // TODO: Support one-dimenstion matmul
            return false;
        }
        auto outputDes = TensorUtils::getDescribe(output);
        // Fill output by zero if one of inputs is empty.
        if (input0->elementSize() == 0 || input1->elementSize() == 0) {
            cmd.command.clear();
            cmd.extras.clear();
            outputDes->regions.clear();
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            return true;
        }
        if (outputs[0]->dimensions() == 2) {
            // Don't change
            return true;
        }
        // Broadcast matmul don't support bias
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
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
        auto o0Dim = output->dimensions();
        int input0_end1 = input0->length(input0->dimensions()-2);
        int input0_end0 = input0->length(input0->dimensions()-1);
        int input1_end1 = input1->length(input1->dimensions()-2);
        int input1_end0 = input1->length(input1->dimensions()-1);
        int e = input0_end1;
        int l = input0_end0;
        int h = input1_end0;
        if (transposeA) {
            e = input0_end0;
            l = input0_end1;
        }
        if (transposeB) {
            h = input1_end1;
        }
        // Compute BroastCast Dims
        auto dimOffset = o0Dim - 2;
        const int maxDimensions = dimOffset;
        int outputStrides[MNN_MAX_TENSOR_DIM];
        int input0Strides[MNN_MAX_TENSOR_DIM];
        int input1Strides[MNN_MAX_TENSOR_DIM];
        auto i0Offset = output->dimensions() - input0->dimensions();
        auto i1Offset = output->dimensions() - input1->dimensions();
        int totalSize = 1;
        int i0Size = 1;
        int i1Size = 1;
        for (int i = maxDimensions - 1; i >=0 ; --i) {
            outputStrides[i] = totalSize;
            input0Strides[i] = 0;
            input1Strides[i] = 0;
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
        auto param = cmd.command[0]->op->main_as_LoopParam();
        ((flatbuffers::Table*)param)->SetField(LoopParam::VT_LOOPNUMBER, totalSize, 0);
        auto rgCmd = param->commands()->GetAs<RegionCommand>(0);
        auto size = (int*)(rgCmd->size()->data());
        size[0] = e; size[1] = l; size[2] = h;
        auto step = (int*)rgCmd->steps()->data();
        step[0] = e * h; step[1] = e * l; step[2] = l * h;
        if (i0Size == 1) {
            step[1] = 0;
        }
        if (i1Size == 1) {
            step[2] = 0;
        }
        // Update view
        {
            auto cStride = (int*)(rgCmd->view()->GetAs<View>(0)->stride()->data());
            cStride[0] = h;//Don't need change others
            auto aStride = (int*)(rgCmd->view()->GetAs<View>(1)->stride()->data());
            if (transposeA) {
                aStride[1] = e;
            } else {
                aStride[0] = l;
            }
            auto bStride = (int*)(rgCmd->view()->GetAs<View>(2)->stride()->data());
            if (transposeB) {
                bStride[2] = l;
            } else {
                bStride[1] = h;
            }
            // don't need change bias's stride
        }
        return true;
    }

    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs, Context& context, CommandBuffer& res) const override {
        bool transposeA = false;
        bool transposeB = false;
        if (op->type() == OpType_BatchMatMul) {
            auto param = op->main_as_BatchMatMulParam();
            transposeA = param->adjX();
            transposeB = param->adjY();
        } else {
            auto param = op->main_as_MatMul();
            transposeA = param->transposeA();
            transposeB = param->transposeB();
        }

        auto input0          = inputs[0];
        auto input1          = inputs[1];
        Tensor* bias         = nullptr;
        auto output          = outputs[0];
        if (inputs.size() > 2) {
            bias = inputs[2];
        }
        auto outputDes = TensorUtils::getDescribe(output);
        // Fill output by zero if one of inputs is empty.
        if (input0->elementSize() == 0 || input1->elementSize() == 0) {
            outputDes->regions.clear();
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            return true;
        }
        int outputNeedSqueeze = 0;
        bool eInsert = false;
        bool hInsert = false;
        if (input0->dimensions() < 2) {
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(input0, newTensor.get(), true);
            newTensor->buffer().type = input0->buffer().type;
            newTensor->buffer().dimensions = 2;
            newTensor->setLength(0, 1);
            newTensor->setLength(1, input0->length(0));
            TensorUtils::getDescribe(newTensor.get())->regions = {TensorUtils::makeFullSlice(input0)};
            TensorUtils::getDescribe(newTensor.get())->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            input0 = newTensor.get();
            res.extras.emplace_back(newTensor);
            outputNeedSqueeze++;
            eInsert = true;
        }
        if (input1->dimensions() < 2) {
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(input1, newTensor.get(), true);
            newTensor->buffer().type = input1->buffer().type;
            newTensor->buffer().dimensions = 2;
            newTensor->setLength(0, input1->length(0));
            newTensor->setLength(1, 1);
            TensorUtils::getDescribe(newTensor.get())->regions = {TensorUtils::makeFullSlice(input1)};
            TensorUtils::getDescribe(newTensor.get())->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            input1 = newTensor.get();
            res.extras.emplace_back(newTensor);
            outputNeedSqueeze++;
            hInsert = true;
        }
        int input0_end1 = input0->length(input0->dimensions()-2);
        int input0_end0 = input0->length(input0->dimensions()-1);
        int input1_end1 = input1->length(input1->dimensions()-2);
        int input1_end0 = input1->length(input1->dimensions()-1);
        int e = input0_end1;
        int l = input0_end0;
        int h = input1_end0;
        if (transposeA) {
            e = input0_end0;
            l = input0_end1;
        }
        if (transposeB) {
            h = input1_end1;
        }
        if (outputNeedSqueeze > 0) {
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(output, newTensor.get(), true);
            newTensor->buffer().dimensions = output->dimensions() + outputNeedSqueeze;
            newTensor->setLength(newTensor->dimensions() - 1, e);
            newTensor->setLength(newTensor->dimensions() - 2, h);
            newTensor->buffer().type = output->buffer().type;
            outputDes->regions = {TensorUtils::makeFullSlice(newTensor.get())};
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            res.extras.emplace_back(newTensor);

            output = newTensor.get();
            outputDes = TensorUtils::getDescribe(output);
        }

        if (output->dimensions() == 2) {
            // Use normal MatMul
            std::shared_ptr<Command> cmd(new Command);
            cmd->op      = op;
            if (bias == nullptr) {
                cmd->inputs  = {input0, input1};
            } else {
                cmd->inputs  = {input0, input1, bias};
            }
            cmd->outputs = {output};
            res.command.emplace_back(cmd);
            return true;
        }
        // Broadcast matmul don't support bias
        // Split MatMul
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
        auto o0Dim = output->dimensions();
        // Compute BroastCast Dims
        auto dimOffset = o0Dim - 2;
        const int maxDimensions = dimOffset;
        int outputStrides[MNN_MAX_TENSOR_DIM];
        int input0Strides[MNN_MAX_TENSOR_DIM];
        int input1Strides[MNN_MAX_TENSOR_DIM];
        auto i0Offset = output->dimensions() - input0->dimensions();
        auto i1Offset = output->dimensions() - input1->dimensions();
        int totalSize = 1;
        int i0Size = 1;
        int i1Size = 1;
        for (int i = maxDimensions - 1; i >=0 ; --i) {
            outputStrides[i] = totalSize;
            input0Strides[i] = 0;
            input1Strides[i] = 0;
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
        flatbuffers::FlatBufferBuilder builder;

        // Create Region Command
        std::vector<flatbuffers::Offset<View>> allViews(3);
        int size[] = {e, l, h};
        int steps[] = {e*h, e*l, l*h, 0};
        auto sizeOffset = builder.CreateVector(size, 3);
        {
            int stride[] = {h, 0, 1};
            auto strideOffset = builder.CreateVector(stride, 3);
            ViewBuilder viewB(builder);
            viewB.add_offset(0);
            viewB.add_stride(strideOffset);
            allViews[0] = viewB.Finish();
        }
        {
            int stride[3];
            stride[2] = 0;
            if (transposeA) {
                stride[0] = 1;
                stride[1] = e;
            } else {
                stride[1] = 1;
                stride[0] = l;
            }
            auto strideOffset = builder.CreateVector(stride, 3);
            ViewBuilder viewB(builder);
            viewB.add_offset(0);
            viewB.add_stride(strideOffset);
            allViews[1] = viewB.Finish();
        }
        {
            int stride[3];
            stride[0] = 0;
            if (transposeB) {
                stride[1] = 1;
                stride[2] = l;
            } else {
                stride[1] = h;
                stride[2] = 1;
            }
            auto strideOffset = builder.CreateVector(stride, 3);
            ViewBuilder viewB(builder);
            viewB.add_offset(0);
            viewB.add_stride(strideOffset);
            allViews[2] = viewB.Finish();
        }
        if (bias != nullptr) {
            int stride[3] = {0, 0, 1};
            auto strideOffset = builder.CreateVector(stride, 3);
            ViewBuilder viewB(builder);
            viewB.add_offset(0);
            viewB.add_stride(strideOffset);
            allViews.emplace_back(viewB.Finish());
        }
        flatbuffers::Offset<flatbuffers::String> nameOffset;
        if (nullptr != op->name()) {
            nameOffset = builder.CreateString(op->name()->c_str());
        }
        MatMulBuilder matMulParam(builder);
        matMulParam.add_transposeA(transposeA);
        matMulParam.add_transposeB(transposeB);
        auto matMulParamOffset = matMulParam.Finish();
        OpBuilder matMulOp(builder);
        matMulOp.add_type(OpType_MatMul);
        matMulOp.add_main(matMulParamOffset.Union());
        matMulOp.add_main_type(OpParameter_MatMul);
        auto opOffset = matMulOp.Finish();
        bool fastway = (i0Size == i1Size) || (i0Size == 1) || (i1Size == 1);
        if (fastway) {
            int inputNumber = 2;
            if (bias != nullptr) {
                inputNumber = 3;
            }
            if (1 == i0Size) {
                steps[1] = 0;
            }
            if (1 == i1Size) {
                steps[2] = 0;
            }
            int number = inputNumber + 1;
            auto viewOffset = builder.CreateVector<flatbuffers::Offset<View>>(allViews);
            int indexes[] = {2, 0, 1, 3};
            int iterIndexes[] = {-1, -1, -1, -1};
            auto indexOffset = builder.CreateVector(indexes, number);
            auto iterIndexesOffset = builder.CreateVector(iterIndexes, number);
            auto stepOffset = builder.CreateVector(steps, number);
            RegionCommandBuilder rgCmdBuilder(builder);
            rgCmdBuilder.add_op(opOffset);
            rgCmdBuilder.add_size(sizeOffset);
            rgCmdBuilder.add_view(viewOffset);
            rgCmdBuilder.add_iterIndexes(iterIndexesOffset);
            rgCmdBuilder.add_indexes(indexOffset);
            rgCmdBuilder.add_steps(stepOffset);
            auto regionCommandOffset = rgCmdBuilder.Finish();
            
            int inputIndexes[] = {0, 1, 3};
            auto inputIndexesOffset = builder.CreateVector(inputIndexes, inputNumber);
            int outputIndexes[] = {2};
            auto outputIndexOffset = builder.CreateVector(outputIndexes, 1);
            
            auto cmdOffset = builder.CreateVector(&regionCommandOffset, 1);
            LoopParamBuilder lpBuilder(builder);
            lpBuilder.add_commands(cmdOffset);
            lpBuilder.add_parallel(true);
            lpBuilder.add_inputIndexes(inputIndexesOffset);
            lpBuilder.add_outputIndexes(outputIndexOffset);
            lpBuilder.add_loopNumber(totalSize);
            lpBuilder.add_tensorNumber(number);
            auto lpOffset = lpBuilder.Finish();

            OpBuilder opBuilder(builder);
            opBuilder.add_main(lpOffset.Union());
            opBuilder.add_main_type(OpParameter_LoopParam);
            opBuilder.add_type(OpType_While);
            if (nullptr != op->name()) {
                opBuilder.add_name(nameOffset);
            }
            builder.Finish(opBuilder.Finish());
            if (bias != nullptr) {
                auto cmd = GeometryComputerUtils::makeCommand(builder, {input0, input1, bias}, {output});
                res.command.emplace_back(std::move(cmd));
            } else {
                auto cmd = GeometryComputerUtils::makeCommand(builder, {input0, input1}, {output});
                res.command.emplace_back(std::move(cmd));
            }
            return true;
        }
        auto i0OffsetTensor = context.allocConst(op, {totalSize}, halide_type_of<int>());
        auto i1OffsetTensor = context.allocConst(op, {totalSize}, halide_type_of<int>());
        if (nullptr == i0OffsetTensor || nullptr == i1OffsetTensor) {
            return false;
        }
        // Commpute Offset Index
        auto i0OffsetTensorPtr = i0OffsetTensor->host<int>();
        auto i1OffsetTensorPtr = i1OffsetTensor->host<int>();
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
            i0OffsetTensorPtr[index] = i0Offset;
            i1OffsetTensorPtr[index] = i1Offset;
        }
        int inputNumber = 4;
        if (bias != nullptr) {
            inputNumber = 5;
        }
        int number = inputNumber + 1;
        int rgNumber = number - 2;
        auto viewOffset = builder.CreateVector<flatbuffers::Offset<View>>(allViews);
        int indexes[] = {4, 0, 1, 5};
        int iterIndexes[] = {-1, 2, 3, -1};
        auto indexOffset = builder.CreateVector(indexes, rgNumber);
        auto iterIndexesOffset = builder.CreateVector(iterIndexes, rgNumber);
        auto stepOffset = builder.CreateVector(steps, rgNumber);
        RegionCommandBuilder rgCmdBuilder(builder);
        rgCmdBuilder.add_op(opOffset);
        rgCmdBuilder.add_size(sizeOffset);
        rgCmdBuilder.add_view(viewOffset);
        rgCmdBuilder.add_iterIndexes(iterIndexesOffset);
        rgCmdBuilder.add_indexes(indexOffset);
        rgCmdBuilder.add_steps(stepOffset);
        auto regionCommandOffset = rgCmdBuilder.Finish();
        
        int inputIndexes[] = {0, 1, 2, 3, 5};
        auto inputIndexesOffset = builder.CreateVector(inputIndexes, inputNumber);
        int outputIndexes[] = {4};
        auto outputIndexOffset = builder.CreateVector(outputIndexes, 1);
        
        auto cmdOffset = builder.CreateVector(&regionCommandOffset, 1);
        LoopParamBuilder lpBuilder(builder);
        lpBuilder.add_commands(cmdOffset);
        lpBuilder.add_parallel(true);
        lpBuilder.add_inputIndexes(inputIndexesOffset);
        lpBuilder.add_outputIndexes(outputIndexOffset);
        lpBuilder.add_loopNumber(totalSize);
        lpBuilder.add_tensorNumber(number);
        auto lpOffset = lpBuilder.Finish();

        OpBuilder opBuilder(builder);
        opBuilder.add_main(lpOffset.Union());
        opBuilder.add_main_type(OpParameter_LoopParam);
        opBuilder.add_type(OpType_While);
        if (nullptr != op->name()) {
            opBuilder.add_name(nameOffset);
        }
        builder.Finish(opBuilder.Finish());
        std::vector<Tensor*> inputLoops{input0, input1, i0OffsetTensor.get(), i1OffsetTensor.get()};
        if (nullptr != bias) {
            inputLoops.emplace_back(bias);
        }
        auto cmd = GeometryComputerUtils::makeCommand(builder, inputLoops, {output});
        res.command.emplace_back(std::move(cmd));
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryBatchMatMul);
    GeometryComputer::registerGeometryComputer(comp, {OpType_BatchMatMul, OpType_MatMul}, Runtime::Compiler_Loop);
}

REGISTER_GEOMETRY(GeometryBatchMatMul, _create);

} // namespace MNN
