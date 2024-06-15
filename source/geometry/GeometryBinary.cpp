//
//  GeometryBinary.cpp
//  MNN
//
//  Created by MNN on 2020/05/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvertUtils.hpp"
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "shape/SizeComputer.hpp"
#define MNN_BINARY_LOOP_OPT
namespace MNN {
class GeometryBinary : public GeometryComputer {
public:
    virtual bool onRecompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        if (res.command.size() != 1) {
            return false;
        }
        auto input0     = inputs[0];
        auto input1     = inputs[1];
        auto output     = outputs[0];
        auto inputL0    = TensorUtils::getRawSize(input0);
        auto inputL1    = TensorUtils::getRawSize(input1);
        auto outputSize = TensorUtils::getRawSize(output);
        auto inp0format = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        auto inp1format = TensorUtils::getDescribe(inputs[1])->dimensionFormat;
        auto outFormat  = TensorUtils::getDescribe(output)->dimensionFormat;
        auto cmdP = res.command[0];
        if (cmdP->op->type() != OpType_BinaryOp) {
            return false;
        }
        MNN_ASSERT(0 != inputL1 && 0 != inputL0 && 0 != outputSize);
        //MNN_PRINT("On compute geometry: %d - %d - %d\n", inputL0, inputL1, outputSize);
        if (1 == inputL0 || 1 == inputL1) {
            // Can directly compute
            cmdP->inputs[0] = input0;
            cmdP->inputs[1] = input1;
            return true;
        }
        // Need Broadcast or same shape
        bool input0Broadcast = false;
        bool input1Broadcast = false;
        if (outputSize != inputL0 || inp0format != outFormat ||
            (output->dimensions() != input0->dimensions() && (MNN_DATA_FORMAT_NC4HW4 == outFormat || context.forwardType() == MNN_FORWARD_OPENCL))// OpenCL default format is MNN_DATA_FORMAT_NC4HW4
            ) {
            input0Broadcast = true;
        }
        if (outputSize != inputL1 || inp1format != outFormat ||
                (output->dimensions() != input1->dimensions() && (MNN_DATA_FORMAT_NC4HW4 == outFormat || context.forwardType() == MNN_FORWARD_OPENCL))) {// OpenCL default format is MNN_DATA_FORMAT_NC4HW4
            input1Broadcast = true;
        }
        auto cacheTensor = std::move(res.extras);
        if (input0Broadcast) {
            std::shared_ptr<Tensor> newTensor;
            if (!cacheTensor.empty()) {
                newTensor = cacheTensor[cacheTensor.size() - 1];
                cacheTensor.erase(cacheTensor.begin() + cacheTensor.size() - 1);
            } else {
                newTensor.reset(new Tensor);
            }
            TensorUtils::copyShape(output, newTensor.get(), true);
            newTensor->buffer().type = output->buffer().type;
            ConvertUtils::broadcastto(input0, newTensor.get());
            input0 = newTensor.get();
            res.extras.emplace_back(newTensor);
        }
        if (input1Broadcast) {
            std::shared_ptr<Tensor> newTensor;
            if (!cacheTensor.empty()) {
                newTensor = cacheTensor[cacheTensor.size() - 1];
                cacheTensor.erase(cacheTensor.begin() + cacheTensor.size() - 1);
            } else {
                newTensor.reset(new Tensor);
            }
            TensorUtils::copyShape(output, newTensor.get(), true);
            newTensor->buffer().type = output->buffer().type;
            ConvertUtils::broadcastto(input1, newTensor.get());
            input1 = newTensor.get();
            res.extras.emplace_back(newTensor);
        }
        cmdP->inputs[0] = input0;
        cmdP->inputs[1] = input1;
        return true;
    }

    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input0     = inputs[0];
        auto input1     = inputs[1];
        auto output     = outputs[0];
        auto inputL0    = TensorUtils::getRawSize(input0);
        auto inputL1    = TensorUtils::getRawSize(input1);
        auto outputSize = TensorUtils::getRawSize(output);
        auto inp0format = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        auto inp1format = TensorUtils::getDescribe(inputs[1])->dimensionFormat;
        auto outFormat  = TensorUtils::getDescribe(output)->dimensionFormat;
        MNN_ASSERT(0 != inputL1 && 0 != inputL0 && 0 != outputSize);
        //MNN_PRINT("On compute geometry: %d - %d - %d\n", inputL0, inputL1, outputSize);
        if (1 == inputL0 || 1 == inputL1) {
            // Can directly compute
            std::shared_ptr<Command> cmdP(new Command);
            auto& cmd = *cmdP;
            cmd.op      = op;
            cmd.inputs  = {input0, input1};
            cmd.outputs = std::move(outputs);
            res.command.emplace_back(std::move(cmdP));
            return true;
        }
        // Need Broadcast or same shape
        bool input0Broadcast = false;
        bool input1Broadcast = false;
        if (outputSize != inputL0 || inp0format != outFormat ||
            (output->dimensions() != input0->dimensions() && (MNN_DATA_FORMAT_NC4HW4 == outFormat || context.forwardType() == MNN_FORWARD_OPENCL))// OpenCL default format is MNN_DATA_FORMAT_NC4HW4
            ) {
            input0Broadcast = true;
        }
        if (outputSize != inputL1 || inp1format != outFormat ||
                (output->dimensions() != input1->dimensions() && (MNN_DATA_FORMAT_NC4HW4 == outFormat || context.forwardType() == MNN_FORWARD_OPENCL))) {// OpenCL default format is MNN_DATA_FORMAT_NC4HW4
            input1Broadcast = true;
        }
#ifdef MNN_BINARY_LOOP_OPT
        // One input need broadcast, the other needn't
        bool singleBroadCast = (!(input0Broadcast && input1Broadcast)) && (input0Broadcast || input1Broadcast);
        bool forwardSupportLoop = inp0format == outFormat && inp1format == outFormat && outFormat != MNN_DATA_FORMAT_NC4HW4 && input0->getType().code == halide_type_float && op->main_as_BinaryOp()->activationType() == 0;
        bool openLoop = context.support(Interpreter::GeometryComputeMask::GEOMETRCOMPUTEMASK_USELOOP);
        if (singleBroadCast && forwardSupportLoop && openLoop) {
            // Use Loop instead of broadcast
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(output, newTensor.get(), true);
            newTensor->buffer().type = output->buffer().type;
            int srcIndex = 1;
            int dstIndex = 2;
            if (input0Broadcast) {
                ConvertUtils::broadcastto(input0, newTensor.get());
            } else {
                srcIndex = 2;
                dstIndex = 1;
                ConvertUtils::broadcastto(input1, newTensor.get());
            }
            auto des = TensorUtils::getDescribe(newTensor.get());
            flatbuffers::FlatBufferBuilder builder;
            BinaryOpBuilder binaryOpParamBuilder(builder);
            binaryOpParamBuilder.add_opType(op->main_as_BinaryOp()->opType());
            auto binaryOpParamOffset = binaryOpParamBuilder.Finish();
            OpBuilder cmdOpBuilder(builder);
            cmdOpBuilder.add_type(OpType_BinaryOp);
            cmdOpBuilder.add_main(binaryOpParamOffset.Union());
            cmdOpBuilder.add_main_type(OpParameter_BinaryOp);
            auto cmdOpOffset = cmdOpBuilder.Finish();
            auto iterIndexesOffset = builder.CreateVector(std::vector<int>{-1, -1, -1});
            auto stepOffset = builder.CreateVector(std::vector<int>{0, 0, 0});
            auto indexesOffset = builder.CreateVector(std::vector<int>{2, 0, 1});
            std::vector<flatbuffers::Offset<RegionCommand>> regionCommands;

            for (int i=0; i<des->regions.size(); ++i) {
                auto& reg = des->regions[i];
                auto sizeOffset = builder.CreateVector(reg.size, 3);
                auto dstStride = builder.CreateVector(reg.dst.stride, 3);
                auto srcStride = builder.CreateVector(reg.src.stride, 3);
                std::vector<flatbuffers::Offset<View>> views(3);
                {
                    ViewBuilder dstBuilder(builder);
                    dstBuilder.add_offset(reg.dst.offset);
                    dstBuilder.add_stride(dstStride);
                    views[0] = dstBuilder.Finish();
                    views[dstIndex] = views[0];
                    ViewBuilder srcBuilder(builder);
                    srcBuilder.add_offset(reg.src.offset);
                    srcBuilder.add_stride(srcStride);
                    views[srcIndex] = srcBuilder.Finish();
                }
                auto viewsOffset = builder.CreateVector<flatbuffers::Offset<View>>(views);
                RegionCommandBuilder cmdBuilder(builder);
                cmdBuilder.add_op(cmdOpOffset);
                cmdBuilder.add_view(viewsOffset);
                cmdBuilder.add_size(sizeOffset);
                cmdBuilder.add_steps(stepOffset);
                cmdBuilder.add_iterIndexes(iterIndexesOffset);
                cmdBuilder.add_indexes(indexesOffset);
                
                regionCommands.emplace_back(cmdBuilder.Finish());
            }
            auto rcmdAllOffset = builder.CreateVector<flatbuffers::Offset<RegionCommand>>(regionCommands);
            auto inputIndexesOffset = builder.CreateVector(std::vector<int>{0, 1});
            auto outputIndexesOffset = builder.CreateVector(std::vector<int>{2});
            LoopParamBuilder loopBuilder(builder);
            loopBuilder.add_commands(rcmdAllOffset);
            loopBuilder.add_loopNumber(1);
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
            auto cmd = GeometryComputerUtils::makeCommand(builder, {input0, input1}, outputs);
            res.command.emplace_back(std::move(cmd));
            return true;
        }
#endif
        if (input0Broadcast) {
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(output, newTensor.get(), true);
            newTensor->buffer().type = output->buffer().type;
            ConvertUtils::broadcastto(input0, newTensor.get());
            input0 = newTensor.get();
            res.extras.emplace_back(newTensor);
        }
        if (input1Broadcast) {
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(output, newTensor.get(), true);
            newTensor->buffer().type = output->buffer().type;
            ConvertUtils::broadcastto(input1, newTensor.get());
            input1 = newTensor.get();
            res.extras.emplace_back(newTensor);
        }
        std::shared_ptr<Command> cmdP(new Command);
        auto& cmd = *cmdP;
        cmd.op      = op;
        cmd.inputs  = {input0, input1};
        cmd.outputs = std::move(outputs);
        res.command.emplace_back(std::move(cmdP));
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryBinary);
    GeometryComputer::registerGeometryComputer(comp, {OpType_BinaryOp});
}

REGISTER_GEOMETRY(GeometryBinary, _create);

} // namespace MNN
