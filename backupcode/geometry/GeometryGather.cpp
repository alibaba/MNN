//
//  GeometryGather.cpp
//  MNN
//
//  Created by MNN on 2020/06/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {

class GeometryGather : public DefaultGeometryComputer {
public:
    virtual std::vector<bool> onGetOutputVirtual(const Op* op, const std::vector<Tensor*>& inputs,
                                                 const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 2);
        MNN_ASSERT(1 == outputs.size());
        auto embedding = inputs[0];
        auto indices   = inputs[1];
        auto output    = outputs[0];

        const int firstDimStride = embedding->buffer().dim[0].stride;
        if (TensorUtils::getDescribe(indices)->usage == MNN::Tensor::InsideDescribe::CONSTANT && firstDimStride != 0) {
            std::vector<bool> res(outputs.size(), true);
            return res;
        }
        return std::vector<bool>(outputs.size(), false);
    }

    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto embedding = inputs[0];
        auto indices   = inputs[1];
        auto output    = outputs[0];

        const int firstDimStride = embedding->buffer().dim[0].stride;
        if (TensorUtils::getDescribe(indices)->usage != MNN::Tensor::InsideDescribe::CONSTANT || firstDimStride == 0) {
            Command cmd;
            cmd.op      = op;
            cmd.inputs  = std::move(inputs);
            cmd.outputs = std::move(outputs);
            res.command.emplace_back(std::move(cmd));
            return true;
        }

        auto bytes = embedding->buffer().type.bytes();

        const size_t indicesCount = indices->elementSize();
        const auto limit          = embedding->length(0);
        const int* indicesData    = indices->host<int32_t>();

        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->regions.clear();
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        for (int i = 0; i < indicesCount; i++) {
            if (indicesData[i] < 0 || indicesData[i] > limit) {
                MNN_PRINT("Gather indice error\n");
                return false;
            }

            Tensor::InsideDescribe::Region slice;
            slice.origin        = embedding;
            slice.size[0]       = 1;
            slice.size[1]       = 1;
            slice.size[2]       = firstDimStride;
            slice.src.offset    = firstDimStride * indicesData[i];
            slice.dst.offset    = i * firstDimStride;
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

class GeometryGatherND : public DefaultGeometryComputer {
public:
    virtual std::vector<bool> onGetOutputVirtual(const Op* op, const std::vector<Tensor*>& inputs,
                                                 const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 2);
        MNN_ASSERT(1 == outputs.size());
        auto params  = inputs[0];
        auto indices = inputs[1];
        auto output  = outputs[0];

        int mSliceN    = 1;
        int mSliceSize = 1;
        for (int i = 0; i < indices->dimensions() - 1; ++i) {
            mSliceN *= indices->length(i);
        }
        auto indiceNd = indices->length(indices->dimensions() - 1);
        std::vector<int> mDimsToCount;
        mDimsToCount.resize(indiceNd);
        for (int i = indiceNd; i < params->dimensions(); ++i) {
            mSliceSize *= params->length(i);
        }

        if (TensorUtils::getDescribe(indices)->usage == MNN::Tensor::InsideDescribe::CONSTANT && mSliceSize != 0) {
            std::vector<bool> res(outputs.size(), true);
            return res;
        } else {
            std::vector<bool> res(outputs.size(), false);
            return res;
        }
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
        std::vector<int> mDimsToCount;
        mDimsToCount.resize(indiceNd);
        for (int i = indiceNd; i < params->dimensions(); ++i) {
            mSliceSize *= params->length(i);
        }

        if (TensorUtils::getDescribe(indice)->usage != MNN::Tensor::InsideDescribe::CONSTANT || mSliceSize == 0) {
            Command cmd;
            cmd.op      = op;
            cmd.inputs  = std::move(inputs);
            cmd.outputs = std::move(outputs);
            res.command.emplace_back(std::move(cmd));
            return true;
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

class GeometryGatherV2 : public DefaultGeometryComputer {
public:
    virtual std::vector<bool> onGetOutputVirtual(const Op* op, const std::vector<Tensor*>& inputs,
                                                 const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() >= 2);
        MNN_ASSERT(1 == outputs.size());
        auto params  = inputs[0];
        auto indices = inputs[1];
        auto output  = outputs[0];

        int axis = 0;
        if (inputs.size() == 3) {
            const Tensor* axisTensor = inputs[2];
            axis                     = axisTensor->host<int32_t>()[0];
        }

        MNN_ASSERT(axis > -params->buffer().dimensions && axis < params->buffer().dimensions);

        if (axis < 0) {
            axis = params->buffer().dimensions + axis;
        }
        const int gatherDimSize = params->buffer().dim[axis].extent;
        const int N             = indices->elementSize();
        MNN_ASSERT(gatherDimSize <= std::numeric_limits<int32_t>::max());

        int inside = 1;
        for (int i = axis + 1; i < params->dimensions(); ++i) {
            inside *= params->length(i);
        }

        if (TensorUtils::getDescribe(indices)->usage == MNN::Tensor::InsideDescribe::CONSTANT && inside != 0) {
            std::vector<bool> res(outputs.size(), true);
            return res;
        }
        return std::vector<bool>(outputs.size(), false);
    }

    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(inputs.size() >= 2);
        MNN_ASSERT(1 == outputs.size());
        auto params  = inputs[0];
        auto indices = inputs[1];
        auto output  = outputs[0];

        int axis = 0;
        if (inputs.size() == 3) {
            const Tensor* axisTensor = inputs[2];
            axis                     = axisTensor->host<int32_t>()[0];
        }
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

        if (TensorUtils::getDescribe(indices)->usage != MNN::Tensor::InsideDescribe::CONSTANT || inside == 0) {
            Command cmd;
            cmd.op      = op;
            cmd.inputs  = std::move(inputs);
            cmd.outputs = std::move(outputs);
            res.command.emplace_back(std::move(cmd));
            return true;
        }

        const int limit               = params->length(axis);
        auto bytes                    = output->buffer().type.bytes();
        const int insideStride        = inside;
        const int outputOutsideStride = inside * N;
        const int inputOutsideStride  = inside * inputs[0]->length(axis);
        const int* indicesPtr         = indices->host<int32_t>();

        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->regions.clear();
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        for (int o = 0; o < outside; ++o) {
            for (int i = 0; i < N; i++) {
                if (indicesPtr[i] < 0 || indicesPtr[i] > limit) {
                    continue;
                }
                Tensor::InsideDescribe::Region slice;
                slice.origin        = params;
                slice.size[0]       = 1;
                slice.size[1]       = 1;
                slice.size[2]       = insideStride;
                slice.src.offset    = inputOutsideStride * o + insideStride * indicesPtr[i];
                slice.dst.offset    = outputOutsideStride * o + i * insideStride;
                slice.src.stride[0] = 1;
                slice.src.stride[1] = 1;
                slice.src.stride[2] = 1;
                slice.dst.stride[0] = 1;
                slice.dst.stride[1] = 1;
                slice.dst.stride[2] = 1;
                outputDes->regions.emplace_back(std::move(slice));
            }
        }
        return true;
    }
};

static void _create() {
//    std::shared_ptr<GeometryComputer> comp(new GeometryGather);
//    GeometryComputer::registerGeometryComputer(comp, {OpType_Gather});
//
//    std::shared_ptr<GeometryComputer> comp2(new GeometryGatherND);
//    GeometryComputer::registerGeometryComputer(comp2, {OpType_GatherND});
//
//    std::shared_ptr<GeometryComputer> comp3(new GeometryGatherV2);
//    GeometryComputer::registerGeometryComputer(comp3, {OpType_GatherV2});
}

REGISTER_GEOMETRY(GeometryGather, _create);

} // namespace MNN
