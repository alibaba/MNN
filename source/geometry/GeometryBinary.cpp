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
namespace MNN {
class GeometryBinary : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input0     = inputs[0];
        auto input1     = inputs[1];
        auto output     = outputs[0];
        auto inputL0    = input0->elementSize();
        auto inputL1    = input1->elementSize();
        auto outputSize = output->elementSize();
        auto inp0format = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        auto inp1format = TensorUtils::getDescribe(inputs[1])->dimensionFormat;
        auto outFormat  = TensorUtils::getDescribe(output)->dimensionFormat;
        MNN_ASSERT(0 != inputL1 && 0 != inputL0 && 0 != outputSize);
        //MNN_PRINT("On compute geometry: %d - %d - %d\n", inputL0, inputL1, outputSize);
        if (1 == inputL0 || 1 == inputL1) {
            // Can directly compute
            Command cmd;
            cmd.op      = op;
            cmd.inputs  = {input0, input1};
            cmd.outputs = std::move(outputs);
            res.command.emplace_back(std::move(cmd));
            return true;
        }
        // Need Broadcast or same shape
        bool input0Broadcast = false;
        bool input1Broadcast = false;
        if (outputSize != inputL0 || inp0format != outFormat ||
            (output->dimensions() != input0->dimensions() && MNN_DATA_FORMAT_NC4HW4 == outFormat)
            ) {
            input0Broadcast = true;
        }
        if (outputSize != inputL1 || inp1format != outFormat ||
                (output->dimensions() != input1->dimensions() && MNN_DATA_FORMAT_NC4HW4 == outFormat)) {
            input1Broadcast = true;
        }
        if (input0Broadcast || input1Broadcast) {
            if ((context.forwardType() == MNN_FORWARD_CPU || context.forwardType() == MNN_FORWARD_CPU_EXTENSION) && inp0format == outFormat && inp1format == outFormat && outFormat != MNN_DATA_FORMAT_NC4HW4 && input0->getType().code == halide_type_float) {
                if (!(input0Broadcast && input1Broadcast)) {
//                if (false) {
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
                    std::unique_ptr<OpT> loopOp(new OpT);
                    loopOp->type = OpType_While;
                    loopOp->main.value = new LoopParamT;
                    loopOp->main.type = OpParameter_LoopParam;
                    auto loop = loopOp->main.AsLoopParam();
                    loop->parallel = false;
                    loop->tensorNumber = 3;
                    loop->inputIndexes = {0, 1};
                    loop->outputIndexes = {2};
                    loop->loopNumber = 1;
                    loop->commands.resize(des->regions.size());
                    for (int i=0; i<loop->commands.size(); ++i) {
                        auto& reg = des->regions[i];
                        loop->commands[i].reset(new RegionCommandT);
                        auto rcmd = loop->commands[i].get();
                        rcmd->size = {reg.size[0], reg.size[1], reg.size[2]};
                        rcmd->indexes = {2, 0, 1};
                        rcmd->iterIndexes = {-1, -1, -1};
                        rcmd->steps = {0, 0, 0};
                        rcmd->view.resize(3);
                        rcmd->view[0].reset(new ViewT);
                        rcmd->view[1].reset(new ViewT);
                        rcmd->view[2].reset(new ViewT);
                        rcmd->op.reset(op->UnPack());
                        rcmd->view[0]->offset = reg.dst.offset;
                        rcmd->view[0]->stride = {reg.dst.stride[0], reg.dst.stride[1], reg.dst.stride[2]};
                        rcmd->view[dstIndex]->offset = reg.dst.offset;
                        rcmd->view[dstIndex]->stride = {reg.dst.stride[0], reg.dst.stride[1], reg.dst.stride[2]};
                        rcmd->view[srcIndex]->offset = reg.src.offset;
                        rcmd->view[srcIndex]->stride = {reg.src.stride[0], reg.src.stride[1], reg.src.stride[2]};
                    }
                    flatbuffers::FlatBufferBuilder builder;
                    if (nullptr != op->name()) {
                        loopOp->name = op->name()->str();
                    }
                    builder.Finish(Op::Pack(builder, loopOp.get()));
                    auto cmd = GeometryComputerUtils::makeCommand(builder, {input0, input1}, outputs);
                    res.command.emplace_back(std::move(cmd));
                    return true;
                }
            }
        }
        if (input0Broadcast) {
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(output, newTensor.get(), true);
            newTensor->buffer().type = output->buffer().type;
            ConvertUtils::broadcastto(input0, newTensor.get());
            input0 = newTensor.get();
            res.extras.emplace_back(newTensor);
        }
        if (outputSize != inputL1 || inp1format != outFormat ||
                (output->dimensions() != input1->dimensions() && MNN_DATA_FORMAT_NC4HW4 == outFormat)) {
            std::shared_ptr<Tensor> newTensor(new Tensor);
            TensorUtils::copyShape(output, newTensor.get(), true);
            newTensor->buffer().type = output->buffer().type;
            ConvertUtils::broadcastto(input1, newTensor.get());
            input1 = newTensor.get();
            res.extras.emplace_back(newTensor);
        }
        Command cmd;
        cmd.op      = op;
        cmd.inputs  = {input0, input1};
        cmd.outputs = std::move(outputs);
        res.command.emplace_back(std::move(cmd));
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryBinary);
    GeometryComputer::registerGeometryComputer(comp, {OpType_BinaryOp});
}

REGISTER_GEOMETRY(GeometryBinary, _create);

} // namespace MNN
