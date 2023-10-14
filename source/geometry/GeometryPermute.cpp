//
//  GeometryPermute.cpp
//  MNN
//
//  Created by MNN on 2020/04/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
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
        int shape[MNN_MAX_TENSOR_DIM];
        if (op->type() == OpType_Permute) {
            auto shapeValue = op->main_as_Permute()->dims();
            if (nullptr != shapeValue) {
                for (int i = 0; i < input->buffer().dimensions; ++i) {
                    shape[i] = shapeValue->data()[i];
                }
            } else {
                for (int i = 0; i < input->buffer().dimensions; ++i) {
                    shape[i] = input->buffer().dimensions - i - 1;
                }
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
                // Fuse dimension if possible
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
        /** Move max three inputShapeSize to last three location. 
         * Don't change max three number relative position
         * */
	bool isReorderShape = false;
	isReorderShape = (inputShapeSize > 4);
        if (inputShapeSize == 4) {
            // TODO: Opt this logic
	    isReorderShape = (inputShape[0] > inputShape[1] + inputShape[2] + inputShape[3]); 
        }
        if (isReorderShape) {
            int max1 = inputShape[0], max2 = -1, max3 = -1;
            // Find Max Three Number
            for (int i = 1; i < inputShapeSize; i++) {
                if (inputShape[i] > max1) {
                    max3 = max2;
                    max2 = max1;
                    max1 = inputShape[i];
                } else if (inputShape[i] > max2) {
                    max3 = max2;
                    max2 = inputShape[i];
                }
                else if (inputShape[i] > max3) {
                    max3 = inputShape[i];
                }
            }
            
            // Move Max Three Number to Last Location
            int lastIndex = inputShapeSize-1;
            for (int i = inputShapeSize-1; i >= 0; i--) {
                if (inputShape[i] == max1) {
                    if(i != lastIndex) {
                        std::swap(inputShape[i], inputShape[lastIndex]);
                        std::swap(inputStrides[i], inputStrides[lastIndex]);
                        std::swap(outputStrides[i], outputStrides[lastIndex]);
                    }
                    max1 = -1;
                    lastIndex--;
                } else if (inputShape[i] == max2) {
                    if(i != lastIndex) {
                        std::swap(inputShape[i], inputShape[lastIndex]);
                        std::swap(inputStrides[i], inputStrides[lastIndex]);
                        std::swap(outputStrides[i], outputStrides[lastIndex]);
                    }
                    max2 = -1;
                    lastIndex--;
                } else if (inputShape[i] == max3) {
                    if(i != lastIndex) {
                        std::swap(inputShape[i], inputShape[lastIndex]);
                        std::swap(inputStrides[i], inputStrides[lastIndex]);
                        std::swap(outputStrides[i], outputStrides[lastIndex]);
                    }
                    max3 = -1;
                    lastIndex--;
                }
                if(lastIndex < inputShapeSize-3) {
                    break;
                }
            }
        }
	// Compute inside, outside, axis
        int inside        = 1;
        int insideStride  = 0;
        int outside       = 1;
        int outsideStride = 0;
        int axis          = 1;
        int axisStride    = 0;
        int breakAxis     = -1;
        int remainSize    = 1;
        int outputInsideStride = 0;
        int outputAxisStride = 0;
        int outputOutsideStride = 0;
        {
            if (inputShapeSize >= 1) {
                inside       = inputShape[inputShapeSize-1];
                insideStride = inputStrides[inputShapeSize-1];
                outputInsideStride = outputStrides[inputShapeSize-1];
            }
            if (inputShapeSize >= 2) {
                axis       = inputShape[inputShapeSize-2];
                axisStride = inputStrides[inputShapeSize-2];
                outputAxisStride = outputStrides[inputShapeSize-2];
            }
            if (inputShapeSize >= 3) {
                outside       = inputShape[inputShapeSize-3];
                outsideStride = inputStrides[inputShapeSize-3];
                outputOutsideStride = outputStrides[inputShapeSize-3];
                breakAxis     = inputShapeSize - 3;
                for (int i = 0; i < inputShapeSize - 3; ++i) {
                    remainSize *= inputShape[i];
                }
            }
        }
        outputDes->regions.resize(remainSize);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        int32_t mod[MNN_MAX_TENSOR_DIM];
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
            int outputOffset = 0;
            for (int i = 0; i < breakAxis; ++i) {
                auto coordinate = value / mod[i];
                inputOffset += coordinate * inputStrides[i];
                outputOffset += coordinate * outputStrides[i];
                value = value % mod[i];
            }
            Tensor::InsideDescribe::Region& slice = outputDes->regions[indice];
            slice.src.offset                      = inputOffset;
            slice.src.stride[0]                   = outsideStride;
            slice.size[0]                         = outside;
            slice.src.stride[1]                   = axisStride;
            slice.size[1]                         = axis;
            slice.src.stride[2]                   = insideStride;
            slice.size[2]                         = inside;
            slice.origin                          = originTensor;
            slice.dst.offset                      = outputOffset;
            slice.dst.stride[0]                   = outputOutsideStride;
            slice.dst.stride[1]                   = outputAxisStride;
            slice.dst.stride[2]                   = outputInsideStride;
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
