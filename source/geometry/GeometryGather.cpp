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
    const int limit = 3;
    if (TensorUtils::getDescribe(indices)->usage == Tensor::InsideDescribe::CONSTANT && N < limit) {
        // Use Raster instead of loop
        auto outDes = TensorUtils::getDescribe(output);
        outDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        outDes->regions.clear();
        outDes->regions.reserve(N);
        auto indicePtr = indices->host<int>();
        auto axisLen = params->length(axis);
        for (int i=0; i<N; ++i) {
            if (indicePtr[i] < 0 || indicePtr[i] >= axisLen) {
                continue;
            }
            Tensor::InsideDescribe::Region reg;
            reg.origin = inputs[0];
            reg.size[0] = 1;
            reg.size[1] = outside;
            reg.size[2] = inside;
            reg.src.offset = indicePtr[i] * inside;
            reg.src.stride[0] = 0;
            reg.src.stride[1] = inside * axisLen;
            reg.src.stride[2] = 1;
            reg.dst.offset = i * inside;
            reg.dst.stride[1] = inside * N;
            reg.dst.stride[2] = 1;
            outDes->regions.emplace_back(std::move(reg));
        }
        return;
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
        auto view0Stride = (int*)rgcmd->view()->GetAs<View>(0)->stride()->data();
        view0Stride[0] = inside * N;
        view0Stride[1] = inside;
        auto view1Stride = (int*)rgcmd->view()->GetAs<View>(1)->stride()->data();
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
        P_broadcastStride = 2,
        P_mulIndice = 3,
        P_indiceOneLine = 4,
        P_MAX
    };
    static bool buildGatherND(const Op* op, Tensor* params, Tensor* indice, Tensor* output,
                              int N, int D, int S, Context& context, CommandBuffer& res, int B) {
        int paramSize = 1;
        for (int i=B; i<params->dimensions(); ++i) {
            paramSize *= params->length(i);
        }
        std::array<std::shared_ptr<Tensor>, 5> midTensors;
        std::shared_ptr<Tensor> constStride(Tensor::createDevice<int>({D}));
        if (!context.allocTensor(constStride.get())) {
            return false;
        }
        midTensors[P_constStride] = constStride;
        for (int i=0; i<D; ++i) {
            int dimCount = paramSize / params->length(i + B);
            constStride->host<int>()[i] = dimCount;
            paramSize = dimCount;
        }
        std::shared_ptr<Tensor> reshapeIndice(Tensor::createDevice<int>({N, D}));
        midTensors[P_reshapeIndice] = reshapeIndice;
        {
            auto des = TensorUtils::getDescribe(reshapeIndice.get());
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->regions = {GeometryComputerUtils::makeRawAddressRef(indice, 0, N * D)};
        }
        std::shared_ptr<Tensor> broadcastStride(Tensor::createDevice<int>({N, D}));
        midTensors[P_broadcastStride] = broadcastStride;
        {
            // [D] => [N, D]
            auto des = TensorUtils::getDescribe(broadcastStride.get());
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->regions.resize(1);
            des->regions[0].origin = constStride.get();
            des->regions[0].size[0] = 1;
            des->regions[0].size[1] = N;
            des->regions[0].size[2] = D;
            des->regions[0].dst.stride[0] = D*N;
            des->regions[0].dst.stride[1] = D;
            des->regions[0].dst.stride[2] = 1;
            des->regions[0].src.stride[0] = 0;
            des->regions[0].src.stride[1] = 0;
            des->regions[0].src.stride[2] = 1;
        }
        std::shared_ptr<Tensor> mulIndice(Tensor::createDevice<int>({N, D}));
        midTensors[P_mulIndice] = mulIndice;
        {
            // [N, D] * [N, D] => [N, D]
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, reshapeIndice.get(), broadcastStride.get(), mulIndice.get());
            res.command.emplace_back(std::move(cmd));
        }
        std::shared_ptr<Tensor> indiceOneLine(Tensor::createDevice<int>({N, 1}));
        midTensors[P_indiceOneLine] = indiceOneLine;
        {
            // [N, D] => [N, 1]
            auto cmd = GeometryComputerUtils::makeReduce(ReductionType_SUM, mulIndice.get(), indiceOneLine.get());
            res.command.emplace_back(std::move(cmd));
        }
        flatbuffers::FlatBufferBuilder builder;
        OpBuilder unaryOp(builder);
        unaryOp.add_type(OpType_UnaryOp);
        auto unaryOpPffset = unaryOp.Finish();
        auto iterIndexesOffset = builder.CreateVector(std::vector<int>{-1, 1});
        auto stepOffset = builder.CreateVector(std::vector<int>{S, 1});
        auto indexesOffset = builder.CreateVector(std::vector<int>{2, 0});
        auto sizeOffset = builder.CreateVector(std::vector<int>{1, 1, S});
        // View 0
        auto view0Stride = builder.CreateVector(std::vector<int>{S, S, 1});
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
        loopBuilder.add_loopNumber(N);
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
        auto cmd = GeometryComputerUtils::makeCommand(builder, {params, indiceOneLine.get()}, {output});
        TensorUtils::getDescribe(output)->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
        res.command.emplace_back(std::move(cmd));
        res.extras.insert(res.extras.end(), midTensors.begin(), midTensors.end());
        return true;
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
        int batchDim = 0;
        if (nullptr != op->main_as_Axis()) {
            batchDim = op->main_as_Axis()->axis();
        }

        for (int i = 0; i < indice->dimensions() - 1; ++i) {
            mSliceN *= indice->length(i);
        }
        auto indiceNd = indice->length(indice->dimensions() - 1);
        for (int i = indiceNd + batchDim; i < params->dimensions(); ++i) {
            mSliceSize *= params->length(i);
        }
        int paramSize = 1;
        for (int i=batchDim; i<params->dimensions(); ++i) {
            paramSize *= params->length(i);
        }
        auto constStride = cmd.extras[P_constStride];
        auto reshapeIndice = cmd.extras[P_reshapeIndice];
        auto broadcastStride = cmd.extras[P_broadcastStride];
        auto mulIndice = cmd.extras[P_mulIndice];
        auto indiceOneLine = cmd.extras[P_indiceOneLine];
        // Set length
        bool needAlloc = constStride->length(0) < indiceNd;
        constStride->setLength(0, indiceNd);
        reshapeIndice->setLength(0, mSliceN);
        reshapeIndice->setLength(1, indiceNd);
        broadcastStride->setLength(0, mSliceN);
        broadcastStride->setLength(1, indiceNd);
        mulIndice->setLength(0, mSliceN);
        mulIndice->setLength(1, indiceNd);
        indiceOneLine->setLength(0, mSliceN);
        indiceOneLine->setLength(1, 1);

        if (needAlloc) {
            if (!context.allocTensor(constStride.get())) {
                return false;
            }
        }
        for (int i=0; i<indiceNd; ++i) {
            int dimCount = paramSize / params->length(i + batchDim);
            constStride->host<int>()[i] = dimCount;
            paramSize = dimCount;
        }
        // recompute reshape
        auto des = TensorUtils::getDescribe(reshapeIndice.get());
        des->extra.offset = 0;
        des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        des->regions = {GeometryComputerUtils::makeRawAddressRef(indice, 0, mSliceN * indiceNd)};
        // recompute broadcast
        des = TensorUtils::getDescribe(broadcastStride.get());
        des->extra.offset = 0;
        des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        des->regions[0].origin = constStride.get();
        des->regions[0].size[0] = 1;
        des->regions[0].size[1] = mSliceN;
        des->regions[0].size[2] = indiceNd;
        des->regions[0].dst.stride[0] = indiceNd*mSliceN;
        des->regions[0].dst.stride[1] = indiceNd;
        des->regions[0].dst.stride[2] = 1;
        // recompute loop
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
        int batchDim = 0;
        if (nullptr != op->main_as_Axis()) {
            batchDim = op->main_as_Axis()->axis();
        }

        int N = 1;
        int S = 1;
        for (int i = 0; i < indice->dimensions() - 1; ++i) {
            N *= indice->length(i);
        }
        auto D = indice->length(indice->dimensions() - 1);
        for (int i = D + batchDim; i < params->dimensions(); ++i) {
            S *= params->length(i);
        }
        return buildGatherND(op, params, indice, output, N, D, S, context, res, batchDim);
    }
};

class GeometryGatherElements : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto data     = inputs[0];
        auto indices  = inputs[1];
        auto output   = outputs[0];
        int axis      = 0;
        if (inputs.size() >= 3) {
            auto axisTensor = inputs[2];
            axis = axisTensor->host<int>()[0];
        }
        auto D  = data->buffer().dimensions;
        auto N  = indices->elementSize();
        if (axis < 0) {
            axis = D + axis;
        }
        // flatten indices/update
        std::shared_ptr<Tensor> flattenIndice(Tensor::createDevice<int>({N}));
        {
            auto ides = TensorUtils::getDescribe(flattenIndice.get());
            ides->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            ides->regions = {GeometryComputerUtils::makeRawAddressRef(indices, 0, N)};
            res.extras.emplace_back(flattenIndice);
        }
        // reindex
        std::shared_ptr<Tensor> newIndice(Tensor::createDevice<int>({N, D}));
        {
            auto des = TensorUtils::getDescribe(newIndice.get());
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->regions.resize(D);
            for (int i = 0; i < D; i++) {
                if (i == axis) {
                    des->regions[i].origin = flattenIndice.get();
                } else {
                    int inner = 1, outter = 1, middle = indices->shape()[i];
                    for (int j = 0; j < i; j++) outter *= indices->shape()[j];
                    for (int j = i + 1; j < D; j++) inner *= indices->shape()[j];
                    MNN_ASSERT(N == inner * middle * outter);
                    auto subIndice = context.allocConst(op, {N}, halide_type_of<int>());
                    auto ptr = subIndice->host<int>();
                    int idx = 0;
                    for (int out = 0; out < outter; out++) {
                        for (int mid = 0; mid < middle; mid++) {
                            for (int in = 0; in < inner; in++) {
                                ptr[idx++] = mid;
                            }
                        }
                    }
                    des->regions[i].origin = subIndice.get();
                }
                des->regions[i].size[2] = N;
                des->regions[i].dst.stride[2] = D;
                des->regions[i].dst.offset = i;
            }
            res.extras.emplace_back(newIndice);
        }
        return GeometryGatherND::buildGatherND(op, data, newIndice.get(), output, N, D, 1, context, res, 0);
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryGather);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Gather, OpType_GatherV2}, Runtime::Compiler_Loop);
    std::shared_ptr<GeometryComputer> comp2(new GeometryGatherND);
    GeometryComputer::registerGeometryComputer(comp2, {OpType_GatherND}, Runtime::Compiler_Loop);
    std::shared_ptr<GeometryComputer> comp3(new GeometryGatherElements);
    GeometryComputer::registerGeometryComputer(comp3, {OpType_GatherElements}, Runtime::Compiler_Loop);
}

REGISTER_GEOMETRY(GeometryGather, _create);

} // namespace MNN
