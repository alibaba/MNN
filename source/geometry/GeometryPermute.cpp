//
//  GeometryPermute.cpp
//  MNN
//
//  Created by MNN on 2020/04/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
class GeometryPermute : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input      = inputs[0];
        auto output     = outputs[0];
        auto inputDes   = TensorUtils::getDescribe(input);
        auto outputDes  = TensorUtils::getDescribe(output);
        auto inputSlice = inputDes->regions;
        MNN_ASSERT(input->dimensions() >= 1);
        MNN_ASSERT(output->dimensions() == input->dimensions());
        auto originTensor = input;
        int basicOffset   = 0;
        std::vector<int> inputStrides(input->buffer().dimensions);
        std::vector<int> shape(input->buffer().dimensions);
        if (op->type() == OpType_Permute) {
            auto shapeValue = op->main_as_Permute()->dims();
            for (int i = 0; i < shape.size(); ++i) {
                shape[i] = shapeValue->data()[i];
            }
        } else if (op->type() == OpType_Transpose) {
            auto shapeValue = inputs[1]->host<int32_t>();
            for (int i = 0; i < shape.size(); ++i) {
                shape[i] = shapeValue[i];
            }
        } else {
            MNN_ASSERT(false);
        }
        int eleSize = 1;
        {
            int stride = 1;
            for (int i = input->buffer().dimensions - 1; i >= 0; --i) {
                inputStrides[i] = stride;
                stride *= input->length(i);
            }
            eleSize = stride;
        }
        // Select not zero dims
        std::vector<int> seperateDimIndexes;
        std::vector<int> outputStrides(input->buffer().dimensions);
        for (int i = 0; i < shape.size(); ++i) {
            outputStrides[i] = inputStrides[shape[i]];
            if (1 != output->length(i)) {
                seperateDimIndexes.emplace_back(i);
            }
        }
        int basicStride = 1;
        // Compute inside, outside, axis
        int inside        = 1;
        int insideStride  = 0;
        int outside       = 1;
        int outsideStride = 0;
        int axis          = 1;
        int axisStride    = 0;
        int breakAxis     = -1;
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
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        std::vector<int32_t> mod(breakAxis + 1);
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
                inputOffset += coordinate * outputStrides[index];
                value = value % mod[i];
            }
            Tensor::InsideDescribe::Region& slice = outputDes->regions[indice];
            slice.src.offset                      = inputOffset + basicOffset;
            slice.src.stride[0]                   = outsideStride * basicStride;
            slice.size[0]                         = outside;
            slice.src.stride[1]                   = axisStride * basicStride;
            slice.size[1]                         = axis;
            slice.src.stride[2]                   = insideStride * basicStride;
            slice.size[2]                         = inside;
            slice.origin                          = originTensor;
            slice.dst.offset                      = indice * outside * axis * inside;
            slice.dst.stride[0]                   = axis * inside;
            slice.dst.stride[1]                   = inside;
            slice.dst.stride[2]                   = 1;
        }
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryPermute);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Transpose, OpType_Permute});
}

REGISTER_GEOMETRY(GeometryPermute, _create);
}; // namespace MNN
