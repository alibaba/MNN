//
//  GeometryScatter.cpp
//  MNN
//
//  Created by MNN on 2022/06/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {
static bool buildScatterND(const Op* op, Tensor* indices, Tensor* updates, Tensor* data, Tensor* output,
                          int N, int D, int S, int totalSize, int reduction,
                          GeometryComputer::Context& context, CommandBuffer& res) {
    // get stride
    std::shared_ptr<Tensor> constStride(Tensor::createDevice<int>({D}));
    if (!context.allocTensor(constStride.get())) {
        return false;
    }
    int count = output->elementSize();
    for (int i = 0; i < D; ++i) {
        count = count / output->length(i);
        constStride->host<int>()[i] = count;
    }
    res.extras.emplace_back(constStride);
    std::shared_ptr<Tensor> broadcastStride(Tensor::createDevice<int>({N, D}));
    {
        // [D] => [N, D]
        auto des = TensorUtils::getDescribe(broadcastStride.get());
        des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        des->regions.resize(1);
        des->regions[0].origin = constStride.get();
        des->regions[0].size[0] = 1;
        des->regions[0].size[1] = N;
        des->regions[0].size[2] = D;
        des->regions[0].dst.stride[0] = N*D;
        des->regions[0].dst.stride[1] = D;
        des->regions[0].dst.stride[2] = 1;
        des->regions[0].src.stride[0] = 0;
        des->regions[0].src.stride[1] = 0;
        des->regions[0].src.stride[2] = 1;
        res.extras.emplace_back(broadcastStride);
    }
    // reshape indices: [dims1, D] -> [N, D]
    std::shared_ptr<Tensor> reshapeIndice(Tensor::createDevice<int>({N, D}));
    {
        auto des = TensorUtils::getDescribe(reshapeIndice.get());
        des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        des->regions = {GeometryComputerUtils::makeRawAddressRef(indices, 0, N * D)};
        res.extras.emplace_back(reshapeIndice);
    }
    // get index
    std::shared_ptr<Tensor> mulIndice(Tensor::createDevice<int>({N, D}));
    {
        // [N, D] * [N, D] => [N, D]
        auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, reshapeIndice.get(), broadcastStride.get(), mulIndice.get());
        res.extras.emplace_back(mulIndice);
        res.command.emplace_back(std::move(cmd));
    }
    std::shared_ptr<Tensor> indiceOneLine(Tensor::createDevice<int>({N, 1}));
    {
        // [N, D] => [N, 1]
        auto cmd = GeometryComputerUtils::makeReduce(ReductionType_SUM, mulIndice.get(), indiceOneLine.get());
        res.extras.emplace_back(indiceOneLine);
        res.command.emplace_back(std::move(cmd));
    }
    auto outputDes = TensorUtils::getDescribe(output);
    flatbuffers::FlatBufferBuilder builder;
    {
        flatbuffers::Offset<Op> loopOpOffset;
        OpBuilder unaryOp(builder);
        unaryOp.add_type(OpType_UnaryOp);
        loopOpOffset = unaryOp.Finish();
        auto iterIndexesOffset = builder.CreateVector(std::vector<int>{1, -1});
        auto stepOffset = builder.CreateVector(std::vector<int>{1, S});
        auto indexesOffset = builder.CreateVector(std::vector<int>{3, 0});
        auto sizeOffset = builder.CreateVector(std::vector<int>{1, 1, S});
        // View 0
        auto view0Stride = builder.CreateVector(std::vector<int>{S, S, 1});
        ViewBuilder view0Builder(builder);
        view0Builder.add_offset(0);
        view0Builder.add_stride(view0Stride);
        auto view0Offset = view0Builder.Finish();
        std::vector<flatbuffers::Offset<View>> views {view0Offset, view0Offset};
        // view0 and view1 is the same
        auto viewAllOffset = builder.CreateVector<flatbuffers::Offset<View>>(views);
        RegionCommandBuilder rcmdBuild(builder);
        rcmdBuild.add_op(loopOpOffset);
        rcmdBuild.add_view(viewAllOffset);
        rcmdBuild.add_indexes(indexesOffset);
        rcmdBuild.add_iterIndexes(iterIndexesOffset);
        rcmdBuild.add_steps(stepOffset);
        rcmdBuild.add_size(sizeOffset);
        rcmdBuild.add_fuse(reduction);
        auto rcmdOffset = rcmdBuild.Finish();
        auto rcmdAllOffset = builder.CreateVector<flatbuffers::Offset<RegionCommand>>({rcmdOffset});
        auto inputIndexesOffset = builder.CreateVector(std::vector<int>{0, 1, 2});
        auto outputIndexesOffset = builder.CreateVector(std::vector<int>{3});

        // init View 0
        auto initindexesOffset = builder.CreateVector(std::vector<int>{3, 2});
        auto initsizeOffset = builder.CreateVector(std::vector<int>{1, 1, totalSize});
        auto initview0Stride = builder.CreateVector(std::vector<int>{totalSize, totalSize, 1});
        ViewBuilder initview0Builder(builder);
        initview0Builder.add_offset(0);
        initview0Builder.add_stride(initview0Stride);
        auto initview0Offset = initview0Builder.Finish();
        auto initview1Offset = initview0Offset;
        if (data->dimensions() == 0) {
            auto initview1Stride = builder.CreateVector(std::vector<int>{0, 0, 0});
            ViewBuilder initview1Builder(builder);
            initview1Builder.add_offset(0);
            initview1Builder.add_stride(initview1Stride);
            initview1Offset = initview1Builder.Finish();
        }
        // view0 and view1 is the same
        auto initviewAllOffset = builder.CreateVector<flatbuffers::Offset<View>>({initview0Offset, initview1Offset});
        RegionCommandBuilder initrcmdBuild(builder);
        initrcmdBuild.add_op(loopOpOffset);
        initrcmdBuild.add_view(initviewAllOffset);
        initrcmdBuild.add_indexes(initindexesOffset);
        initrcmdBuild.add_size(initsizeOffset);
        auto initrcmdOffset = initrcmdBuild.Finish();
        auto initrcmdOffsetMulti = builder.CreateVector<flatbuffers::Offset<RegionCommand>>({initrcmdOffset});
        LoopParamBuilder loopBuilder(builder);
        loopBuilder.add_initCommand(initrcmdOffsetMulti);
        loopBuilder.add_commands(rcmdAllOffset);
        loopBuilder.add_parallel(false);
        loopBuilder.add_loopNumber(N);
        loopBuilder.add_tensorNumber(4);
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
    auto cmd = GeometryComputerUtils::makeCommand(builder, {updates, indiceOneLine.get(), data}, {output});
    TensorUtils::getDescribe(output)->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
    res.command.emplace_back(std::move(cmd));
    return true;
}
class GeometryScatterNd : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        /*
         ScatterNd do below operation:
            indices = [dims1, D]
            updates = [dims1, dims2]
            output  = [dims3]
            assert(len(dims1) + len(dims2) = len(dims3))
            output = np.copy(data)
            update_indices = indices.shape[:-1]
            for idx in np.ndindex(update_indices):
                output[indices[idx]] = updates[idx]
         So:
            indices = [N, D]
            updates = [N, S]
            output  = [X, S]
            stride  = [s_1, s_2, ..., s_D]
            index   = sum(indices * stride) = [N, 1]
            for i in range(N):
                output[index[i]] = updates[i]
         */
        auto indices  = inputs[0];
        auto updates  = inputs[1];
        auto shape    = inputs[2];
        int reduction = op->main_as_BinaryOp() ? op->main_as_BinaryOp()->opType() : -1;
        Tensor* data  = nullptr;
        if (inputs.size() == 4) {
            data = inputs[3];
        } else {
            auto type = updates->getType();
            data = context.allocConst(op, {}, type).get();
            memset(data->host<void>(), 0, type.bytes());
        }
        auto output  = outputs[0];
        auto totalSize = output->elementSize();
        int N = 1;
        for (int i = 0; i < indices->dimensions() - 1; ++i) {
            N *= indices->length(i);
        }
        auto D = indices->length(indices->dimensions() - 1);
        int S = 1;
        for (int i = D; i < updates->dimensions(); ++i) {
            S *= updates->length(i);
        }
        if (N == 0 || S == 0) {
            auto outputDes = TensorUtils::getDescribe(output);
            outputDes->regions = {TensorUtils::makeFullSlice(data)};
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            return true;
        }
        return buildScatterND(op, indices, updates, data, output, N, D, S, totalSize, reduction, context, res);
    }
};

class GeometryScatterElements : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto param    = op->main_as_BinaryOp();
        int reduction = param->opType();
        auto data     = inputs[0];
        auto indices  = inputs[1];
        auto updates  = inputs[2];
        auto output   = outputs[0];
        int axis = 0;
        if (inputs.size() >= 4) {
            axis = inputs[3]->host<int>()[0];
        }
        auto D  = data->buffer().dimensions;
        auto N  = indices->elementSize();
        if (axis < 0) {
            axis = D + axis;
        }
        if (N == 0) {
            auto outputDes = TensorUtils::getDescribe(output);
            outputDes->regions = {TensorUtils::makeFullSlice(data)};
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            return true;
        }
        // flatten indices/update
        std::shared_ptr<Tensor> flattenIndice(Tensor::createDevice<int>({N}));
        std::shared_ptr<Tensor> flattenUpdate(Tensor::createDevice({N}, updates->getType(), Tensor::TENSORFLOW));
        {
            auto ides = TensorUtils::getDescribe(flattenIndice.get());
            ides->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            ides->regions = {GeometryComputerUtils::makeRawAddressRef(indices, 0, N)};
            res.extras.emplace_back(flattenIndice);
            auto udes = TensorUtils::getDescribe(flattenUpdate.get());
            udes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            udes->regions = {GeometryComputerUtils::makeRawAddressRef(updates, 0, N)};
            res.extras.emplace_back(flattenUpdate);
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
        return buildScatterND(op, newIndice.get(), flattenUpdate.get(), data, output, N, D, 1, output->elementSize(), reduction, context, res);
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryScatterNd);
    GeometryComputer::registerGeometryComputer(comp, {OpType_ScatterNd}, Runtime::Compiler_Loop);
    std::shared_ptr<GeometryComputer> comp1(new GeometryScatterElements);
    GeometryComputer::registerGeometryComputer(comp1, {OpType_ScatterElements}, Runtime::Compiler_Loop);
}

REGISTER_GEOMETRY(GeometryScatter, _create);

} // namespace MNN
