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
    flatbuffers::FlatBufferBuilder builder;
    OpBuilder unaryOp(builder);
    unaryOp.add_type(OpType_UnaryOp);
    auto unaryOpPffset = unaryOp.Finish();
    auto iterIndexesOffset = builder.CreateVector(std::vector<int>{-1, 1});
    auto stepOffset = builder.CreateVector(std::vector<int>{inside, inside});
    auto indexesOffset = builder.CreateVector(std::vector<int>{2, 0});
    auto sizeOffset = builder.CreateVector(std::vector<int>{outside, 1, inside});
    // View 0
    auto view0Stride = builder.CreateVector(std::vector<int>{inside * N, inside, 1});
    ViewBuilder view0Builder(builder);
    view0Builder.add_offset(0);
    view0Builder.add_stride(view0Stride);
    auto view0Offset = view0Builder.Finish();
    // View1
    auto view1Stride = builder.CreateVector(std::vector<int>{inside * params->length(axis), inside, 1});
    ViewBuilder view1Builder(builder);
    view1Builder.add_offset(0);
    view1Builder.add_stride(view1Stride);
    auto view1Offset = view1Builder.Finish();
    auto viewAllOffset = builder.CreateVector<flatbuffers::Offset<View>>({view0Offset, view1Offset});

    RegionCommandBuilder rcmdBuild(builder);
    rcmdBuild.add_op(unaryOpPffset);
    rcmdBuild.add_view(viewAllOffset);
    rcmdBuild.add_indexes(indexesOffset);
    rcmdBuild.add_iterIndexes(iterIndexesOffset);
    rcmdBuild.add_steps(stepOffset);
    rcmdBuild.add_size(sizeOffset);
    auto rcmdOffset = rcmdBuild.Finish();
    auto rcmdAllOffset = builder.CreateVector<flatbuffers::Offset<RegionCommand>>({rcmdOffset});
    auto inputIndexesOffset = builder.CreateVector(std::vector<int>{0, 1});
    auto outputIndexesOffset = builder.CreateVector(std::vector<int>{2});
    LoopParamBuilder loopBuilder(builder);
    loopBuilder.add_commands(rcmdAllOffset);
    loopBuilder.add_loopNumber(indices->elementSize());
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
    virtual bool onRecompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                             Context& context, CommandBuffer& cmd) const override {
        if (cmd.command.size() != 1) {
            return false;
        }
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
        auto loopCmd = cmd.command[0];
        auto param = loopCmd->op->main_as_LoopParam();
        // Reset parameters for last command
        ((flatbuffers::Table*)param)->SetField(LoopParam::VT_LOOPNUMBER, indices->elementSize(), 0);
        auto rgcmd = param->commands()->GetAs<RegionCommand>(0);
        auto step = (int*)rgcmd->steps()->data();
        step[0] = inside;
        step[1] = inside;
        auto size = (int*)rgcmd->size()->data();
        size[0] = outside;
        size[2] = inside;
        auto view0Stride = (int*)rgcmd->view()->GetAs<View>(0)->stride();
        view0Stride[0] = inside * N;
        view0Stride[1] = inside;
        auto view1Stride = (int*)rgcmd->view()->GetAs<View>(1)->stride();
        view1Stride[0] = inside * params->length(axis);
        view1Stride[1] = inside;
        return true;
    }
};

class GeometryGatherND : public GeometryComputer {
public:
    enum MID_POSITION {
        P_constStride = 0,
        P_reshapeIndice = 1,
        P_reshapeIndiceFloat = 2,
        P_indiceFloat = 3,
        P_indiceOneLine = 4,
        P_MAX
    };
    static void makeLoopCommand(flatbuffers::FlatBufferBuilder& builder, int mSliceSize, int mSliceN, const Op* op) {
        OpBuilder unaryOp(builder);
        unaryOp.add_type(OpType_UnaryOp);
        auto unaryOpPffset = unaryOp.Finish();
        auto iterIndexesOffset = builder.CreateVector(std::vector<int>{-1, 1});
        auto stepOffset = builder.CreateVector(std::vector<int>{mSliceSize, 1});
        auto indexesOffset = builder.CreateVector(std::vector<int>{2, 0});
        auto sizeOffset = builder.CreateVector(std::vector<int>{1, 1, mSliceSize});
        // View 0
        auto view0Stride = builder.CreateVector(std::vector<int>{mSliceSize, mSliceSize, 1});
        ViewBuilder view0Builder(builder);
        view0Builder.add_offset(0);
        view0Builder.add_stride(view0Stride);
        auto view0Offset = view0Builder.Finish();
        // view0 and view1 is the same
        auto viewAllOffset = builder.CreateVector<flatbuffers::Offset<View>>({view0Offset, view0Offset});

        RegionCommandBuilder rcmdBuild(builder);
        rcmdBuild.add_op(unaryOpPffset);
        rcmdBuild.add_view(viewAllOffset);
        rcmdBuild.add_indexes(indexesOffset);
        rcmdBuild.add_iterIndexes(iterIndexesOffset);
        rcmdBuild.add_steps(stepOffset);
        rcmdBuild.add_size(sizeOffset);
        auto rcmdOffset = rcmdBuild.Finish();
        auto rcmdAllOffset = builder.CreateVector<flatbuffers::Offset<RegionCommand>>({rcmdOffset});
        auto inputIndexesOffset = builder.CreateVector(std::vector<int>{0, 1});
        auto outputIndexesOffset = builder.CreateVector(std::vector<int>{2});
        LoopParamBuilder loopBuilder(builder);
        loopBuilder.add_commands(rcmdAllOffset);
        loopBuilder.add_loopNumber(mSliceN);
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
    }
    virtual bool onRecompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                             Context& context, CommandBuffer& cmd) const override {
        if (cmd.extras.size() != P_MAX) {
            return false;
        }
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
        for (int i = indiceNd; i < params->dimensions(); ++i) {
            mSliceSize *= params->length(i);
        }
        auto paramSize = params->elementSize();
        auto constStride = cmd.extras[P_constStride];
        auto reshapeIndice = cmd.extras[P_reshapeIndice];
        auto reshapeIndiceFloat = cmd.extras[P_reshapeIndiceFloat];
        auto indiceFloat = cmd.extras[P_indiceFloat];
        auto indiceOneLine = cmd.extras[P_indiceOneLine];
        // Set length
        bool needAlloc = constStride->length(0) < indiceNd;
        constStride->setLength(0, indiceNd);
        reshapeIndice->setLength(0, mSliceN);
        reshapeIndice->setLength(1, indiceNd);
        reshapeIndiceFloat->setLength(0, mSliceN);
        reshapeIndiceFloat->setLength(1, indiceNd);
        indiceFloat->setLength(0, mSliceN);
        indiceOneLine->setLength(0, mSliceN);

        if (needAlloc) {
            if (!context.allocTensor(constStride.get())) {
                return false;
            }
        }
        for (int i=0; i<indiceNd; ++i) {
            int dimCount = paramSize / params->length(i);
            constStride->host<float>()[i] = (float)dimCount;
            paramSize = dimCount;
        }
        reshapeIndice->buffer().device = 0;
        reshapeIndice->buffer().host = 0;
        auto des = TensorUtils::getDescribe(reshapeIndice.get());
        des->extra.offset = 0;
        des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        des->backend = nullptr;
        des->regions = {GeometryComputerUtils::makeRawAddressRef(indice, 0, mSliceN * indiceNd)};
        
        auto loopCmd = cmd.command[cmd.command.size() - 1];
        auto param = loopCmd->op->main_as_LoopParam();
        // Reset parameters for last command
        ((flatbuffers::Table*)param)->SetField(LoopParam::VT_LOOPNUMBER, mSliceN, 0);
        auto rgCmd = param->commands()->GetAs<RegionCommand>(0);
        auto stepData = (int*)rgCmd->steps()->data();
        stepData[0] = mSliceSize;
        auto sizeData = (int*)rgCmd->size()->data();
        sizeData[2] = mSliceSize;
        auto strideData = (int*)rgCmd->view()->GetAs<View>(0)->stride()->data();
        strideData[0] = mSliceSize;
        strideData[1] = mSliceSize;
        strideData = (int*)rgCmd->view()->GetAs<View>(1)->stride()->data();
        strideData[0] = mSliceSize;
        strideData[1] = mSliceSize;
        return true;
    }

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
        for (int i = indiceNd; i < params->dimensions(); ++i) {
            mSliceSize *= params->length(i);
        }
        auto paramSize = params->elementSize();
        std::array<std::shared_ptr<Tensor>, 5> midTensors;
        std::shared_ptr<Tensor> constStride(Tensor::createDevice<int>({indiceNd, 1}));
        if (!context.allocTensor(constStride.get())) {
            return false;
        }
        midTensors[P_constStride] = constStride;
        for (int i=0; i<indiceNd; ++i) {
            int dimCount = paramSize / params->length(i);
            constStride->host<float>()[i] = (float)dimCount;
            paramSize = dimCount;
        }
        std::shared_ptr<Tensor> reshapeIndice(Tensor::createDevice<int>({mSliceN, indiceNd}));
        midTensors[P_reshapeIndice] = reshapeIndice;
        {
            auto des = TensorUtils::getDescribe(reshapeIndice.get());
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->regions = {GeometryComputerUtils::makeRawAddressRef(indice, 0, mSliceN * indiceNd)};
        }
        std::shared_ptr<Tensor> reshapeIndiceFloat(Tensor::createDevice<float>({mSliceN, indiceNd}));
        midTensors[P_reshapeIndiceFloat] = reshapeIndiceFloat;
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
        }
        std::shared_ptr<Tensor> indiceFloat(Tensor::createDevice<float>({mSliceN, 1}));
        midTensors[P_indiceFloat] = indiceFloat;
        {
            // MatMul
            auto cmd = GeometryComputerUtils::makeMatMul(reshapeIndiceFloat.get(), constStride.get(), indiceFloat.get());
            res.command.emplace_back(std::move(cmd));
        }
        std::shared_ptr<Tensor> indiceOneLine(Tensor::createDevice<int>({mSliceN, 1}));
        midTensors[P_indiceOneLine] = indiceOneLine;
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
        }

        auto indiceData = indice->host<int32_t>();
        auto outputDes = TensorUtils::getDescribe(output);
        flatbuffers::FlatBufferBuilder builder;
        makeLoopCommand(builder, mSliceSize, mSliceN, op);
        auto cmd = GeometryComputerUtils::makeCommand(builder, {params, indiceOneLine.get()}, outputs);
        TensorUtils::getDescribe(output)->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
        res.command.emplace_back(std::move(cmd));
        res.extras.insert(res.extras.end(), midTensors.begin(), midTensors.end());
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryGather);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Gather, OpType_GatherV2}, Runtime::Compiler_Loop);

    std::shared_ptr<GeometryComputer> comp2(new GeometryGatherND);
    GeometryComputer::registerGeometryComputer(comp2, {OpType_GatherND}, Runtime::Compiler_Loop);
}

REGISTER_GEOMETRY(GeometryGather, _create);

} // namespace MNN
