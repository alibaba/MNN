//
//  GeometryStridedSlice.cpp
//  MNN
//
//  Created by MNN on 2020/04/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {
class GeometryStridedSlice : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        Tensor* input = inputs[0];
        // input haven't realized
        auto output     = outputs[0];
        auto outputDes = TensorUtils::getDescribe(output);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;

        auto parameter  = op->main_as_StridedSliceParam();
        Tensor* begin   = inputs[1];
        Tensor* end     = inputs[2];
        Tensor* strided = inputs[3];
        int32_t inputShape[MNN_MAX_TENSOR_DIM];
        int32_t inputStride[MNN_MAX_TENSOR_DIM];
        {
            int stride = 1;
            for (int i = input->buffer().dimensions - 1; i >= 0; --i) {
                inputShape[i]  = input->buffer().dim[i].extent;
                inputStride[i] = stride;
                stride *= inputShape[i];
            }
        }
        int stridedSliceDimension = begin->buffer().dim[0].extent;
        int32_t beginShape[MNN_MAX_TENSOR_DIM];
        int32_t endShape[MNN_MAX_TENSOR_DIM];
        int32_t stridedShape[MNN_MAX_TENSOR_DIM];
        std::vector<int32_t> outputShape;

        int32_t beginMask[MNN_MAX_TENSOR_DIM];
        for (int i = 0; i < stridedSliceDimension; i++) {
            beginMask[i] = parameter->beginMask() & (1 << i);
        }
        int32_t endMask[MNN_MAX_TENSOR_DIM];
        for (int i = 0; i < stridedSliceDimension; i++) {
            endMask[i] = parameter->endMask() & (1 << i);
        }
        int32_t shrinkAxisMask[MNN_MAX_TENSOR_DIM];
        for (int i = 0; i < stridedSliceDimension; i++) {
            shrinkAxisMask[i] = parameter->shrinkAxisMask() & (1 << i);
        }
        auto beginAndEndShapeLimit = [](int shape, int dimSize, bool exclusive) -> int {
            int maxShape = dimSize - 1, minShape = -dimSize;
            if (exclusive) {
                ++maxShape;
                --minShape;
            }
            shape = (shape > maxShape ? maxShape : shape);
            shape = (shape < minShape ? minShape : shape);
            if (shape < 0) {
                shape += dimSize;
            }
            return shape;
        };

        for (int i = 0; i < stridedSliceDimension; i++) {
            if (beginMask[i] > 0) {
                beginShape[i] = 0;
            } else {
                beginShape[i] = std::min(inputShape[i], begin->host<int32_t>()[i]);
            }
            if (beginShape[i] < 0) {
                beginShape[i] += input->buffer().dim[i].extent;
            }
            if (endMask[i] > 0) {
                endShape[i] = inputShape[i];
            } else {
                endShape[i] = beginAndEndShapeLimit(end->host<int32_t>()[i], inputShape[i], true);
            }
            // FIXME: Currently we don't support zero shape, use dimension 0 instead
            if (beginShape[i] == inputShape[i]) {
                return true;
            }
            stridedShape[i] = shrinkAxisMask[i] > 0 ? 1 : strided->host<int32_t>()[i];

            if (shrinkAxisMask[i] == 0) {
                if (stridedShape[i] > 0) {
                    int size = (endShape[i] - beginShape[i] - 1) / stridedShape[i] + 1;
                    outputShape.push_back(size);
                } else {
                    int size = (endShape[i] - beginShape[i] + 1) / stridedShape[i] + 1;
                    outputShape.push_back(size);
                }
            } else {
                outputShape.push_back(1);
            }
        }

        int outputDimensionsWithoutRemain = (int)outputShape.size();
        int dimensionRemained             = input->buffer().dimensions - stridedSliceDimension;
        for (int i = 0; i < dimensionRemained; i++) {
            outputShape.push_back(input->buffer().dim[outputDimensionsWithoutRemain + i].extent);
            stridedShape[stridedSliceDimension + i] = 1;
            beginShape[stridedSliceDimension + i] = 0;
        }
        int remainSize = 1;
        std::vector<int> remainDims;
        for (int i = 0; i < (int)outputShape.size() - 3; ++i) {
            remainSize *= outputShape[i];
            remainDims.emplace_back(outputShape[i]);
        }
        outputDes->regions.resize(remainSize);
        int regionSize        = outputShape.size() < 3 ? outputShape.size() : 3;
        std::vector<int32_t> mod(remainDims.size());
        OpCommonUtils::computeStride(mod.data(), remainDims.data(), (int)remainDims.size());
        int outputStrideTotal = 1;
        int basicInputOffset  = 0;
        for (int i = 0; i < regionSize; ++i) {
            int pos  = outputShape.size() - i - 1;
            auto len = outputShape[pos];
            basicInputOffset += inputStride[pos] * beginShape[pos];
            outputStrideTotal *= len;
        }
        std::vector<int> coordinates(remainSize);
        for (int r = 0; r < remainSize; ++r) {
            OpCommonUtils::unravelIndexHelper(coordinates, mod, mod.size(), r);
            int inputOffset = basicInputOffset;
            for (int i = 0; i < remainDims.size(); ++i) {
                inputOffset += coordinates[i] * inputStride[i] * stridedShape[i];
            }

            auto& reg      = outputDes->regions[r];
            reg.dst.offset = r * outputStrideTotal;
            reg.src.offset = inputOffset;
            reg.origin     = input;
            for (int i = 0; i < regionSize; ++i) {
                int pos                   = outputShape.size() - i - 1;
                reg.size[3 - i - 1]       = outputShape[pos];
                reg.src.stride[3 - i - 1] = inputStride[pos] * stridedShape[pos];
            }
            reg.dst.stride[0] = reg.size[1] * reg.size[2];
            reg.dst.stride[1] = reg.size[2];
            reg.dst.stride[2] = 1;
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryStridedSlice);
    GeometryComputer::registerGeometryComputer(comp, {OpType_StridedSlice});
}

REGISTER_GEOMETRY(GeometryStridedSlice, _create);

} // namespace MNN
