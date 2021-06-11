//
//  GeometryGather.cpp
//  MNN
//
//  Created by MNN on 2020/06/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {
#define MNN_OPEN_GATHER
#ifdef MNN_OPEN_GATHER
static void _computeGather(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           GeometryComputer::Context& context, CommandBuffer& res, const Op* op) {
    int axis = 0;
    if (inputs.size() == 3) {
        const Tensor *axisTensor = inputs[2];
        axis                     = axisTensor->host<int32_t>()[0];
    }
    if (op->main_type() == OpParameter_Axis) {
        axis = op->main_as_Axis()->axis();
    }
    auto params  = inputs[0];
    auto indices = inputs[1];
    auto output  = outputs[0];
    MNN_ASSERT(axis > -params->buffer().dimensions && axis < params->buffer().dimensions);
    if (axis < 0) {
        axis = params->buffer().dimensions + axis;
    }
    const int gatherDimSize = params->buffer().dim[axis].extent;
    const int N             = indices->elementSize();
    MNN_ASSERT(gatherDimSize <= std::numeric_limits<int32_t>::max());

    int inside  = 1;
    int outside = 1;
    for (int i = 0; i < axis; ++i) {
        outside *= params->length(i);
    }
    for (int i = axis + 1; i < params->dimensions(); ++i) {
        inside *= params->length(i);
    }
    std::unique_ptr<OpT> newop(new OpT);
    newop->type = OpType_While;
    newop->main.value = new LoopParamT;
    newop->main.type = OpParameter_LoopParam;
    auto loop = newop->main.AsLoopParam();
    loop->tensorNumber = 3;
    loop->inputIndexes = {0, 1};
    loop->outputIndexes = {2};
    loop->loopNumber = indices->elementSize();
    std::unique_ptr<RegionCommandT> rcmd(new RegionCommandT);
    rcmd->size = {outside, 1, inside};
    rcmd->view.resize(2);
    rcmd->view[1].reset(new ViewT);
    rcmd->view[1]->offset = 0;
    rcmd->view[1]->stride = {inside * params->length(axis), inside, 1};
    rcmd->view[0].reset(new ViewT);
    rcmd->view[0]->offset = 0;
    rcmd->view[0]->stride = {inside * N, inside, 1};
    rcmd->indexes = {2, 0};
    rcmd->steps = {inside, inside};
    rcmd->iterIndexes = {-1, 1};
    rcmd->op.reset(new OpT);
    rcmd->op->type = OpType_UnaryOp;
    loop->commands.emplace_back(std::move(rcmd));
    if (op->name() != nullptr) {
        newop->name = op->name()->str();
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Op::Pack(builder, newop.get()));
    auto cmd = GeometryComputerUtils::makeCommand(builder, {params, indices}, outputs);
    TensorUtils::getDescribe(output)->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
    res.command.emplace_back(std::move(cmd));
}

class GeometryGather : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        _computeGather(inputs, outputs, context, res, op);
        return true;
    }
};

class GeometryGatherND : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto params = inputs[0];
        auto indice = inputs[1];
        auto output = outputs[0];

        int mSliceN    = 1;
        int mSliceSize = 1;
        for (int i = 0; i < indice->dimensions() - 1; ++i) {
            mSliceN *= indice->length(i);
        }
        auto indiceNd = indice->length(indice->dimensions() - 1);
        std::vector<int> mDimsToCount;
        mDimsToCount.resize(indiceNd);
        for (int i = indiceNd; i < params->dimensions(); ++i) {
            mSliceSize *= params->length(i);
        }
        auto paramSize = params->elementSize();
        for (int i = 0; i < indiceNd; ++i) {
            mDimsToCount[i] = paramSize / params->length(i);
            paramSize       = mDimsToCount[i];
        }
        auto constStride = context.allocConst(op, {indiceNd, 1}, halide_type_of<float>());
        for (int i=0; i<indiceNd; ++i) {
            constStride->host<float>()[i] = (float)mDimsToCount[i];
        }
        std::shared_ptr<Tensor> reshapeIndice(Tensor::createDevice<int>({mSliceN, indiceNd}));
        {
            auto des = TensorUtils::getDescribe(reshapeIndice.get());
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->regions = {GeometryComputerUtils::makeRawAddressRef(indice, 0, mSliceN * indiceNd)};
            res.extras.emplace_back(reshapeIndice);
        }
        std::shared_ptr<Tensor> reshapeIndiceFloat(Tensor::createDevice<float>({mSliceN, indiceNd}));
        {
            flatbuffers::FlatBufferBuilder builder;
            CastParamBuilder builder_(builder);
            builder_.add_dstT(DataType_DT_FLOAT);
            auto mainOffset = builder_.Finish().Union();
            OpBuilder opB(builder);
            opB.add_type(OpType_Cast);
            opB.add_main(mainOffset);
            opB.add_main_type(OpParameter_CastParam);
            builder.Finish(opB.Finish());
            auto cmd = GeometryComputerUtils::makeCommand(builder, {reshapeIndice.get()}, {reshapeIndiceFloat.get()});
            res.command.emplace_back(std::move(cmd));
            res.extras.emplace_back(reshapeIndiceFloat);
        }
        std::shared_ptr<Tensor> indiceFloat(Tensor::createDevice<float>({mSliceN, 1}));
        {
            // MatMul
            auto cmd = GeometryComputerUtils::makeMatMul(reshapeIndiceFloat.get(), constStride.get(), indiceFloat.get());
            res.command.emplace_back(std::move(cmd));
            res.extras.emplace_back(indiceFloat);
        }
        std::shared_ptr<Tensor> indiceOneLine(Tensor::createDevice<int>({mSliceN, 1}));
        {
            flatbuffers::FlatBufferBuilder builder;
            CastParamBuilder builder_(builder);
            builder_.add_dstT(DataType_DT_INT32);
            auto mainOffset = builder_.Finish().Union();
            OpBuilder opB(builder);
            opB.add_type(OpType_Cast);
            opB.add_main(mainOffset);
            opB.add_main_type(OpParameter_CastParam);
            builder.Finish(opB.Finish());
            auto cmd = GeometryComputerUtils::makeCommand(builder, {indiceFloat.get()}, {indiceOneLine.get()});
            res.command.emplace_back(std::move(cmd));
            res.extras.emplace_back(indiceOneLine);
        }

        auto indiceData = indice->host<int32_t>();
        auto outputDes = TensorUtils::getDescribe(output);
        std::unique_ptr<OpT> newop(new OpT);
        newop->type = OpType_While;
        newop->main.value = new LoopParamT;
        newop->main.type = OpParameter_LoopParam;
        if (op->name() != nullptr) {
            newop->name = op->name()->str();
        }
        auto loop = newop->main.AsLoopParam();
        loop->tensorNumber = 3;
        loop->inputIndexes = {0, 1};
        loop->outputIndexes = {2};
        loop->loopNumber = mSliceN;
        std::unique_ptr<RegionCommandT> rcmd(new RegionCommandT);
        rcmd->size = {1, 1, mSliceSize};
        rcmd->view.resize(2);
        rcmd->view[0].reset(new ViewT);
        rcmd->view[0]->offset = 0;
        rcmd->view[0]->stride = {mSliceSize, mSliceSize, 1};
        rcmd->view[1].reset(new ViewT);
        rcmd->view[1]->offset = 0;
        rcmd->view[1]->stride = {mSliceSize, mSliceSize, 1};
        rcmd->indexes = {2, 0};
        rcmd->steps = {mSliceSize, 1};
        rcmd->iterIndexes = {-1, 1};
        rcmd->op.reset(new OpT);
        rcmd->op->type = OpType_UnaryOp;
        loop->commands.emplace_back(std::move(rcmd));
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(Op::Pack(builder, newop.get()));
        auto cmd = GeometryComputerUtils::makeCommand(builder, {params, indiceOneLine.get()}, outputs);
        TensorUtils::getDescribe(output)->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
        res.command.emplace_back(std::move(cmd));
        return true;
    }
};
#endif
static void _create() {
#ifdef MNN_OPEN_GATHER
    std::shared_ptr<GeometryComputer> comp(new GeometryGather);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Gather, OpType_GatherV2}, Runtime::Compiler_Loop);

    std::shared_ptr<GeometryComputer> comp2(new GeometryGatherND);
    GeometryComputer::registerGeometryComputer(comp2, {OpType_GatherND}, Runtime::Compiler_Loop);
#endif
}

REGISTER_GEOMETRY(GeometryGather, _create);

} // namespace MNN
