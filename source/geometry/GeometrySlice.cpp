//
//  GeometrySlice.cpp
//  MNN
//
//  Created by MNN on 2020/04/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {
class GeometrySliceTF : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input = inputs[0];
        // these two inputs should be const
        auto begin_tensor = inputs[1];

        auto beginPtr = begin_tensor->host<int32_t>();

        std::vector<int> seperateDimIndexes;
        std::vector<int> outputStrides(input->buffer().dimensions);
        auto output   = outputs[0];
        int stride    = 1;
        int srcOffset = 0;
        for (int i = input->buffer().dimensions - 1; i >= 0; --i) {
            outputStrides[i] = stride;
            auto begin = beginPtr[i];
            if (begin < 0) {
                begin += input->length(i);
            }
            srcOffset += begin * stride;
            stride *= input->length(i);
        }
        for (int i = 0; i < output->buffer().dimensions; ++i) {
            if (1 != output->length(i)) {
                seperateDimIndexes.emplace_back(i);
            }
        }
        auto outputDes  = TensorUtils::getDescribe(output);
        int basicStride = 1;
        // Compute inside, outside, axis
        int inside        = 1;
        int insideStride  = 0;
        int outside       = 1;
        int outsideStride = 0;
        int axis          = 1;
        int axisStride    = 0;
        int breakAxis     = 0;
        int remainSize    = 1;
        {
            if (seperateDimIndexes.size() >= 1) {
                auto index   = seperateDimIndexes[seperateDimIndexes.size() - 1];
                inside       = output->length(index);
                insideStride = outputStrides[index];
            }
            if (seperateDimIndexes.size() >= 2) {
                auto index = seperateDimIndexes[seperateDimIndexes.size() - 2];
                axis       = output->length(index);
                axisStride = outputStrides[index];
            }
            if (seperateDimIndexes.size() >= 3) {
                auto index    = seperateDimIndexes[seperateDimIndexes.size() - 3];
                outside       = output->length(index);
                outsideStride = outputStrides[index];
                breakAxis     = (int)seperateDimIndexes.size() - 3;
                for (int i = 0; i < seperateDimIndexes.size() - 3; ++i) {
                    remainSize *= output->length(seperateDimIndexes[i]);
                }
            }
        }
        outputDes->regions.resize(remainSize);
        std::vector<int32_t> mod(breakAxis);
        for (int i = 0; i < breakAxis; ++i) {
            int value = 1;
            for (int j = i + 1; j < breakAxis; ++j) {
                auto index = seperateDimIndexes[j];
                value *= output->length(index);
            }
            mod[i] = value;
        }
        for (int indice = 0; indice < remainSize; ++indice) {
            int value       = indice;
            int inputOffset = 0;
            for (int i = 0; i < breakAxis; ++i) {
                auto coordinate = value / mod[i];
                auto index      = seperateDimIndexes[i];
                inputOffset += (coordinate)*outputStrides[index];
                value = value % mod[i];
            }
            outputDes->memoryType                 = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            Tensor::InsideDescribe::Region& slice = outputDes->regions[indice];
            slice.src.offset                      = inputOffset + srcOffset;
            slice.src.stride[0]                   = outsideStride * basicStride;
            slice.size[0]                         = outside;
            slice.src.stride[1]                   = axisStride * basicStride;
            slice.size[1]                         = axis;
            slice.src.stride[2]                   = insideStride * basicStride;
            slice.size[2]                         = inside;
            slice.origin                          = input;
            slice.dst.offset                      = indice * outside * axis * inside;
            slice.dst.stride[0]                   = axis * inside;
            slice.dst.stride[1]                   = inside;
            slice.dst.stride[2]                   = 1;
        }
        return true;
    }
};
class GeometrySlice : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input    = inputs[0];
        int axis      = 0;
        bool inputFix = false;
        if (op->type() == OpType_Slice) {
            auto slice = op->main_as_Slice();
            axis       = slice->axis();
        } else if (op->type() == OpType_Unpack) {
            axis     = op->main_as_Axis()->axis();
            inputFix = true;
        }

        if (axis < 0) {
            axis = axis + input->dimensions();
        }
        int outside = 1;
        int inside  = 1;
        for (int i = 0; i < axis; ++i) {
            outside *= input->length(i);
        }
        for (int i = axis + 1; i < input->dimensions(); ++i) {
            inside *= input->length(i);
        }
        auto inputZero = input->elementSize() <= 0;
        int offset = 0;
        for (int i = 0; i < outputs.size(); ++i) {
            auto outputDes = TensorUtils::getDescribe(outputs[i]);
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            if (inputZero) {
                outputDes->regions.clear();
                continue;
            }
            outputDes->regions.resize(1);
            auto& slice           = outputDes->regions[0];
            slice.src.offset      = offset * inside;
            slice.origin          = input;
            slice.size[0]         = outside;
            slice.size[2]         = inside;
            slice.src.stride[0]   = input->length(axis) * inside;
            slice.src.stride[1]   = inside;
            slice.src.stride[2]   = 1;
            if (inputFix) {
                slice.size[1] = 1;
                offset += 1;
            } else {
                slice.size[1] = outputs[i]->length(axis);
                offset += outputs[i]->length(axis);
            }
            slice.dst.offset = 0;
            slice.dst.stride[0] = inside * slice.size[1];
            slice.dst.stride[1] = slice.size[2];
            slice.dst.stride[2] = 1;
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometrySlice);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Slice, OpType_Unpack});
    std::shared_ptr<GeometryComputer> comp2(new GeometrySliceTF);
    GeometryComputer::registerGeometryComputer(comp2, {OpType_SliceTf});
}

REGISTER_GEOMETRY(GeometrySlice, _create);

} // namespace MNN
