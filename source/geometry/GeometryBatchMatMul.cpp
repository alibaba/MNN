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
        Tensor* bias         = nullptr;
        auto output          = outputs[0];
        if (inputs.size() > 2) {
            bias = inputs[2];
        }
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
        std::unique_ptr<OpT> newop(new OpT);
        if (nullptr != op->name()) {
            newop->name = op->name()->str();
        }
        newop->type = OpType_While;
        newop->main.value = new LoopParamT;
        newop->main.type = OpParameter_LoopParam;
        auto loop = newop->main.AsLoopParam();
        loop->parallel = true;
        loop->tensorNumber = 5;
        loop->inputIndexes = {0, 1, 2, 3};
        loop->outputIndexes = {4};
        loop->loopNumber = totalSize;
        std::unique_ptr<RegionCommandT> rcmd(new RegionCommandT);
        rcmd->size = {e, l, h};
        rcmd->view.resize(3);
        rcmd->view[1].reset(new ViewT);
        rcmd->view[1]->offset = 0;
        if (transposeA) {
            rcmd->view[1]->stride = {1, e, 0};
        } else {
            rcmd->view[1]->stride = {l, 1, 0};
        }
        rcmd->view[2].reset(new ViewT);
        rcmd->view[2]->offset = 0;
        if (transposeB) {
            rcmd->view[2]->stride = {0, 1, l};
        } else {
            rcmd->view[2]->stride = {0, h, 1};
        }
        rcmd->view[0].reset(new ViewT);
        rcmd->view[0]->offset = 0;
        rcmd->view[0]->stride = {h, 0, 1};
        rcmd->indexes = {4, 0, 1};// C, A, B
        rcmd->steps = {e*h, e*l, l*h};
        rcmd->iterIndexes = {-1, 2, 3};
        if (bias != nullptr) {
            loop->tensorNumber = 6;
            loop->inputIndexes = {0, 1, 2, 3, 5};
            loop->outputIndexes = {4};
            std::unique_ptr<ViewT> biasView(new ViewT);
            biasView->offset = 0;
            biasView->stride = {0, 0, 1};
            rcmd->view.emplace_back(std::move(biasView));
            rcmd->iterIndexes.emplace_back(-1);
            rcmd->steps.emplace_back(0);
            rcmd->indexes = {4, 0, 1, 5};
        }
        rcmd->op.reset(new OpT);
        rcmd->op->type = OpType_MatMul;
        rcmd->op->main.type = OpParameter_MatMul;
        rcmd->op->main.value = new MatMulT;
        rcmd->op->main.AsMatMul()->transposeB = transposeB;
        rcmd->op->main.AsMatMul()->transposeA = transposeA;
        if (i0Size == i1Size && i0Size == totalSize) {
            // Don't need broadcast
            loop->tensorNumber = 3;
            loop->inputIndexes = {0, 1};
            loop->outputIndexes = {2};
            rcmd->iterIndexes = {-1, -1, -1};
            rcmd->indexes = {2, 0, 1};
            if (bias != nullptr) {
                loop->tensorNumber = 4;
                loop->inputIndexes = {0, 1, 3};
                loop->outputIndexes = {2};
                rcmd->iterIndexes = {-1, -1, -1, -1};
                rcmd->indexes = {2, 0, 1, 3};
            }
            loop->commands.emplace_back(std::move(rcmd));
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(Op::Pack(builder, newop.get()));
            if (bias != nullptr) {
                auto cmd = GeometryComputerUtils::makeCommand(builder, {input0, input1, bias}, outputs);
                res.command.emplace_back(std::move(cmd));
            } else {
                auto cmd = GeometryComputerUtils::makeCommand(builder, {input0, input1}, outputs);
                res.command.emplace_back(std::move(cmd));
            }
            return true;
        }
        loop->commands.emplace_back(std::move(rcmd));
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(Op::Pack(builder, newop.get()));
        auto i0OffsetTensor = context.allocConst(op, {totalSize}, halide_type_of<int>());
        auto i1OffsetTensor = context.allocConst(op, {totalSize}, halide_type_of<int>());
        if (nullptr == i0OffsetTensor || nullptr == i1OffsetTensor) {
            return false;
        }
        // Commpute Offset Index
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
            i0OffsetTensor->host<int>()[index] = i0Offset;
            i1OffsetTensor->host<int>()[index] = i1Offset;
        }
        std::vector<Tensor*> inputLoops{input0, input1, i0OffsetTensor.get(), i1OffsetTensor.get()};
        if (nullptr != bias) {
            inputLoops.emplace_back(bias);
        }
        auto cmd = GeometryComputerUtils::makeCommand(builder, inputLoops, outputs);
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
