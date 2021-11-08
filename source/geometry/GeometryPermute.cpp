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
        int shape[MNN_MAX_TENSOR_DIM];
        if (op->type() == OpType_Permute) {
            auto shapeValue = op->main_as_Permute()->dims();
            for (int i = 0; i < input->buffer().dimensions; ++i) {
                shape[i] = shapeValue->data()[i];
            }
        } else if (op->type() == OpType_Transpose) {
            auto shapeValue = inputs[1]->host<int32_t>();
            for (int i = 0; i < input->buffer().dimensions; ++i) {
                shape[i] = shapeValue[i];
            }
        } else {
            MNN_ASSERT(false);
        }
        int inputShape[MNN_MAX_TENSOR_DIM];
        int inputStrides[MNN_MAX_TENSOR_DIM];
        int inputShapeSize = 0;
        int preAxis = -2;
        for (int i=0; i<input->buffer().dimensions; ++i) {
            auto axis = shape[i];
            auto len = input->length(axis);
            if (1 == len) {
                continue;
            }
            if (axis - preAxis == 1) {
                inputShape[inputShapeSize - 1] *= len;
            } else {
                if (preAxis >= 0) {
                    // Compute last stride
                    int stride = 1;
                    for (int v=preAxis+1; v < input->buffer().dimensions; ++v) {
                        stride *= input->length(v);
                    }
                    inputStrides[inputShapeSize - 1] = stride;
                }
                inputShapeSize+=1;
                inputShape[inputShapeSize - 1] = len;
            }
            preAxis = shape[i];
        }
        if (preAxis >= 0) {
            // Compute last stride
            int stride = 1;
            for (int v=preAxis+1; v < input->buffer().dimensions; ++v) {
                stride *= input->length(v);
            }
            inputStrides[inputShapeSize - 1] = stride;
        }
        if (0 == inputShapeSize) {
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            outputDes->regions = {TensorUtils::makeFullSlice(input)};
            return true;
        }
        int outputStrides[MNN_MAX_TENSOR_DIM];
        {
            int stride = 1;
            for (int i=inputShapeSize-1; i>=0; --i) {
                outputStrides[i] = stride;
                stride *= inputShape[i];
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
            if (inputShapeSize >= 1) {
                inside       = inputShape[inputShapeSize-1];
                insideStride = inputStrides[inputShapeSize-1];
            }
            if (inputShapeSize >= 2) {
                axis       = inputShape[inputShapeSize-2];
                axisStride = inputStrides[inputShapeSize-2];
            }
            if (inputShapeSize >= 3) {
                outside       = inputShape[inputShapeSize-3];
                outsideStride = inputStrides[inputShapeSize-3];
                breakAxis     = inputShapeSize - 3;
                for (int i = 0; i < inputShapeSize - 3; ++i) {
                    remainSize *= inputShape[i];
                }
            }
        }
        outputDes->regions.resize(remainSize);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        std::vector<int32_t> mod(breakAxis + 1);
        for (int i = 0; i < breakAxis; ++i) {
            int value = 1;
            for (int j = i + 1; j < breakAxis; ++j) {
                value *= inputShape[j];
            }
            mod[i] = value;
        }
        for (int indice = 0; indice < remainSize; ++indice) {
            int value       = indice;
            int inputOffset = 0;
            for (int i = 0; i < breakAxis; ++i) {
                auto coordinate = value / mod[i];
                inputOffset += coordinate * inputStrides[i];
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
