//
//  GeometryCumSum.cpp
//  MNN
//
//  Created by MNN on 2020/06/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <numeric>
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {

class GeometryCumSum : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto shape = inputs[0]->shape();
        int axis = (inputs[1]->host<int>()[0] + shape.size()) % shape.size();
        int outside = std::accumulate(shape.begin(), shape.begin() + axis, 1, [](int a, int b) { return a * b; });
        int inside = std::accumulate(shape.begin() + axis + 1, shape.end(), 1, [](int a, int b) { return a * b; });
        bool exclusive = op->main_as_CumSum()->exclusive(), reverse = op->main_as_CumSum()->reverse();

        auto outDes = TensorUtils::getDescribe(outputs[0]);
        outDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        if (!exclusive) {
            outDes->regions.resize(1);
            auto& reg = outDes->regions[0];
            int offset = (reverse ? (shape[axis] - 1) * inside : 0);
            reg.origin = inputs[0];
            reg.src.offset = reg.dst.offset = offset;
            reg.src.stride[0] = reg.dst.stride[0] = inside * shape[axis];
            reg.size[0] = outside;
            reg.size[1] = inside;
        }
        if (shape[axis] == 1) {
            return true;
        }

        flatbuffers::FlatBufferBuilder builder;
        BinaryOpBuilder binaryOpParamBuilder(builder);
        binaryOpParamBuilder.add_opType(BinaryOpOperation_ADD);
        auto binaryOpParamOffset = binaryOpParamBuilder.Finish();
        OpBuilder cmdOpBuilder(builder);
        cmdOpBuilder.add_type(OpType_BinaryOp);
        cmdOpBuilder.add_main(binaryOpParamOffset.Union());
        cmdOpBuilder.add_main_type(OpParameter_BinaryOp);
        auto cmdOpOffset = cmdOpBuilder.Finish();

        auto viewStride = builder.CreateVector(std::vector<int>{shape[axis] * inside, 1, 1});
        int step = inside, offset = inside;
        if (reverse) {
            step = -inside;
            offset = (shape[axis] - 2) * inside;
        }
        std::vector<flatbuffers::Offset<View>> views(3);
        ViewBuilder view0(builder);
        view0.add_stride(viewStride);
        view0.add_offset(offset);
        views[0] = view0.Finish();
        ViewBuilder view1(builder);
        view1.add_stride(viewStride);
        view1.add_offset(offset - step);
        views[1] = view1.Finish();
        views[2] = views[exclusive ? 1 : 0];

        auto viewsOffset = builder.CreateVector<flatbuffers::Offset<View>>(views);
        auto sizeOffset = builder.CreateVector(std::vector<int>{outside, inside, 1});
        auto stepOffset = builder.CreateVector(std::vector<int>{step, step, step});
        auto iterIndexesOffset = builder.CreateVector(std::vector<int>{-1, -1, -1});
        auto indexesOffset = builder.CreateVector(std::vector<int>{2, 0, 1});

        RegionCommandBuilder cmdBuilder(builder);
        cmdBuilder.add_op(cmdOpOffset);
        cmdBuilder.add_view(viewsOffset);
        cmdBuilder.add_size(sizeOffset);
        cmdBuilder.add_steps(stepOffset);
        cmdBuilder.add_iterIndexes(iterIndexesOffset);
        cmdBuilder.add_indexes(indexesOffset);

        std::vector<flatbuffers::Offset<RegionCommand>> regionCommands;
        regionCommands.emplace_back(cmdBuilder.Finish());
        auto rcmdAllOffset = builder.CreateVector<flatbuffers::Offset<RegionCommand>>(regionCommands);
        auto inputIndexesOffset = builder.CreateVector(std::vector<int>{0, 1});
        auto outputIndexesOffset = builder.CreateVector(std::vector<int>{2});
        LoopParamBuilder loopBuilder(builder);
        loopBuilder.add_parallel(false); // cumsum(i) = cumsum(i-1) + x(i), so can't do outside parallel
        loopBuilder.add_commands(rcmdAllOffset);
        loopBuilder.add_loopNumber(shape[axis] - 1);
        loopBuilder.add_tensorNumber(3);
        loopBuilder.add_inputIndexes(inputIndexesOffset);
        loopBuilder.add_outputIndexes(outputIndexesOffset);
        auto loopOffset = loopBuilder.Finish();
        flatbuffers::Offset<flatbuffers::String> nameOffset;
        if (nullptr != op->name()) {
            nameOffset = builder.CreateString(op->name()->c_str());
        }
        OpBuilder finishBuilder(builder);
        finishBuilder.add_main(loopOffset.Union());
        finishBuilder.add_main_type(OpParameter_LoopParam);
        finishBuilder.add_type(OpType_While);
        if (nullptr != op->name()) {
            finishBuilder.add_name(nameOffset);
        }
        builder.Finish(finishBuilder.Finish());
        auto cmd = GeometryComputerUtils::makeCommand(builder, {outputs[0], inputs[0]}, outputs);
        res.command.emplace_back(std::move(cmd));

        return true;
    }
};

class GeometryCumProd : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto shape = inputs[0]->shape();
        int axis = op->main_as_Axis()->axis();
        if (axis < 0) {
            axis += shape.size();
        }
        int outside = std::accumulate(shape.begin(), shape.begin() + axis, 1, [](int a, int b) { return a * b; });
        int inside = std::accumulate(shape.begin() + axis + 1, shape.end(), 1, [](int a, int b) { return a * b; });
        auto outDes = TensorUtils::getDescribe(outputs[0]);
        outDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        outDes->regions.resize(1);
        auto& reg = outDes->regions[0];
        reg.origin = inputs[0];
        reg.src.offset = reg.dst.offset = 0;
        reg.src.stride[0] = reg.dst.stride[0] = inside * shape[axis];
        reg.size[0] = outside;
        reg.size[1] = inside;

        if (shape[axis] == 1) {
            return true;
        }

        flatbuffers::FlatBufferBuilder builder;
        BinaryOpBuilder binaryOpParamBuilder(builder);
        binaryOpParamBuilder.add_opType(BinaryOpOperation_MUL);
        auto binaryOpParamOffset = binaryOpParamBuilder.Finish();
        OpBuilder cmdOpBuilder(builder);
        cmdOpBuilder.add_type(OpType_BinaryOp);
        cmdOpBuilder.add_main(binaryOpParamOffset.Union());
        cmdOpBuilder.add_main_type(OpParameter_BinaryOp);
        auto cmdOpOffset = cmdOpBuilder.Finish();

        auto viewStride = builder.CreateVector(std::vector<int>{shape[axis] * inside, 1, 1});
        int step = inside, offset = inside;
        std::vector<flatbuffers::Offset<View>> views(3);
        ViewBuilder view0(builder);
        view0.add_stride(viewStride);
        view0.add_offset(offset);
        views[0] = view0.Finish();
        ViewBuilder view1(builder);
        view1.add_stride(viewStride);
        view1.add_offset(offset - step);
        views[1] = view1.Finish();
        views[2] = views[0];

        auto viewsOffset = builder.CreateVector<flatbuffers::Offset<View>>(views);
        auto sizeOffset = builder.CreateVector(std::vector<int>{outside, inside, 1});
        auto stepOffset = builder.CreateVector(std::vector<int>{step, step, step});
        auto iterIndexesOffset = builder.CreateVector(std::vector<int>{-1, -1, -1});
        auto indexesOffset = builder.CreateVector(std::vector<int>{2, 0, 1});

        RegionCommandBuilder cmdBuilder(builder);
        cmdBuilder.add_op(cmdOpOffset);
        cmdBuilder.add_view(viewsOffset);
        cmdBuilder.add_size(sizeOffset);
        cmdBuilder.add_steps(stepOffset);
        cmdBuilder.add_iterIndexes(iterIndexesOffset);
        cmdBuilder.add_indexes(indexesOffset);

        std::vector<flatbuffers::Offset<RegionCommand>> regionCommands;
        regionCommands.emplace_back(cmdBuilder.Finish());
        auto rcmdAllOffset = builder.CreateVector<flatbuffers::Offset<RegionCommand>>(regionCommands);
        auto inputIndexesOffset = builder.CreateVector(std::vector<int>{0, 1});
        auto outputIndexesOffset = builder.CreateVector(std::vector<int>{2});
        LoopParamBuilder loopBuilder(builder);
        loopBuilder.add_parallel(false); // cumprod(i) = cumprod(i-1) * x(i), so can't do outside parallel
        loopBuilder.add_commands(rcmdAllOffset);
        loopBuilder.add_loopNumber(shape[axis] - 1);
        loopBuilder.add_tensorNumber(3);
        loopBuilder.add_inputIndexes(inputIndexesOffset);
        loopBuilder.add_outputIndexes(outputIndexesOffset);
        auto loopOffset = loopBuilder.Finish();
        flatbuffers::Offset<flatbuffers::String> nameOffset;
        if (nullptr != op->name()) {
            nameOffset = builder.CreateString(op->name()->c_str());
        }
        OpBuilder finishBuilder(builder);
        finishBuilder.add_main(loopOffset.Union());
        finishBuilder.add_main_type(OpParameter_LoopParam);
        finishBuilder.add_type(OpType_While);
        if (nullptr != op->name()) {
            finishBuilder.add_name(nameOffset);
        }
        builder.Finish(finishBuilder.Finish());
        auto cmd = GeometryComputerUtils::makeCommand(builder, {outputs[0], inputs[0]}, outputs);
        res.command.emplace_back(std::move(cmd));

        return true;
    }
};


static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryCumSum);
    GeometryComputer::registerGeometryComputer(comp, {OpType_CumSum});
    std::shared_ptr<GeometryComputer> comp1(new GeometryCumProd);
    GeometryComputer::registerGeometryComputer(comp1, {OpType_CumProd});
}

REGISTER_GEOMETRY(GeometryCumSum, _create);

}
