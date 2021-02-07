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
    const int limit               = params->length(axis);

    /* Compute Offset Begin*/
    std::shared_ptr<Tensor> offset(Tensor::createDevice<int>({2, N}));
    auto offsetDes = TensorUtils::getDescribe(offset.get());
    offsetDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
    // srcOffset
    offsetDes->regions.emplace_back(GeometryComputerUtils::makeRawAddressRef(indices, 0, N, 0));
    // Compute Dst Offset: Range(0, N, 1)
    Tensor* dstOffset = nullptr;
    {
        auto start = context.allocConst(op, {}, halide_type_of<int>());
        auto delta = context.allocConst(op, {}, halide_type_of<int>());
        std::shared_ptr<Tensor> dstRange(Tensor::createDevice<int>({N}));
        flatbuffers::FlatBufferBuilder builder;
        OpBuilder opB(builder);
        opB.add_type(OpType_Range);
        builder.Finish(opB.Finish());
        start->host<int>()[0] = 0;
        delta->host<int>()[0] = 1;
        res.command.emplace_back(GeometryComputerUtils::makeCommand(builder, {start.get(), start.get(), delta.get()}, {dstRange.get()}));
        res.extras.emplace_back(dstRange);
        offsetDes->regions.emplace_back(GeometryComputerUtils::makeRawAddressRef(dstRange.get(), 0, N, N));
    }
    std::shared_ptr<Tensor> offsetMulStride(Tensor::createDevice<int>({2, N}));
    {
        auto stride = context.allocConst(op, {}, halide_type_of<int>());
        stride->host<int>()[0] = inside;
        res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, offset.get(), stride.get(), offsetMulStride.get()));
    }

    /* Compute Offset End*/
    Tensor::InsideDescribe::Region slice;
    slice.origin        = params;
    slice.size[0]       = outside;
    slice.size[1]       = 1;
    slice.size[2]       = inside;
    slice.src.stride[0] = inside * params->length(axis);
    slice.src.stride[1] = inside;
    slice.src.stride[2] = 1;
    slice.dst.stride[0] = inside * N;
    slice.dst.stride[1] = inside;
    slice.dst.stride[2] = 1;
    slice.offset = offsetMulStride.get();
    res.extras.emplace_back(offset);
    res.extras.emplace_back(offsetMulStride);
    TensorUtils::getDescribe(output)->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
    TensorUtils::getDescribe(output)->regions = {slice};
}

class GeometryGather : public DefaultGeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        _computeGather(inputs, outputs, context, res, op);
        return true;
    }
};

class GeometryGatherV2 : public DefaultGeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        _computeGather(inputs, outputs, context, res, op);
        return true;
    }
};

class GeometryGatherND : public DefaultGeometryComputer {
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
        mDimsToCount.resize(indiceNd);
        auto indiceData = indice->host<int32_t>();

        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->regions.clear();
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        for (int i = 0; i < mSliceN; i++) {
            int fromPos = 0;
            for (int j = 0; j < indiceNd; ++j) {
                fromPos += mDimsToCount[j] * indiceData[i * indiceNd + j];
            }

            Tensor::InsideDescribe::Region slice;
            slice.origin        = params;
            slice.size[0]       = 1;
            slice.size[1]       = 1;
            slice.size[2]       = mSliceSize;
            slice.src.offset    = fromPos;
            slice.dst.offset    = i * mSliceSize;
            slice.src.stride[0] = 1;
            slice.src.stride[1] = 1;
            slice.src.stride[2] = 1;
            slice.dst.stride[0] = 1;
            slice.dst.stride[1] = 1;
            slice.dst.stride[2] = 1;
            outputDes->regions.emplace_back(std::move(slice));
        }
        return true;
    }
};
#endif
static void _create() {
#ifdef MNN_OPEN_GATHER
    std::shared_ptr<GeometryComputer> comp(new GeometryGather);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Gather});

    std::shared_ptr<GeometryComputer> comp2(new GeometryGatherND);
    GeometryComputer::registerGeometryComputer(comp2, {OpType_GatherND});

    std::shared_ptr<GeometryComputer> comp3(new GeometryGatherV2);
    GeometryComputer::registerGeometryComputer(comp3, {OpType_GatherV2});
#endif
}

REGISTER_GEOMETRY(GeometryGather, _create);

} // namespace MNN
