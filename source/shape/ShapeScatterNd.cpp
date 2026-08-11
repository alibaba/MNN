//
//  ShapeScatterNd.cpp
//  MNN
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//


#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
// Size Computer
class ShapeScatterNd : public SizeComputer {
    bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                       const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(3 <= inputs.size());
        auto indices = inputs[0];
        auto updates = inputs[1];
        auto shape   = inputs[2];
        auto output  = outputs[0];
        //MNN_CHECK(shape->dimensions() == 1, "shape rank should be one");
        const int indicesDimension = indices->dimensions();
        const int dimension = shape->length(0);
        // Validate: indices last dim must not exceed output rank.
        // (indices[..., K] requires K <= output rank; K < rank means slice update,
        //  which is legal. Only K > rank is invalid.)
        if (indices->length(indicesDimension - 1) > dimension) {
            MNN_ERROR("ScatterNd: indices last dim (%d) > output rank (%d)\n",
                      indices->length(indicesDimension - 1), dimension);
            return false;
        }

        const int outerDims = indicesDimension - 1;
        // Validate: updates outer dims must match indices outer dims
        int indicesOuterSize = 1;
        for (int i = 0; i < outerDims; ++i) {
            indicesOuterSize *= indices->length(i);
        }
        if (updates->elementSize() < indicesOuterSize) {
            MNN_ERROR("ScatterNd: updates size (%d) < indices outer size (%d)\n",
                      updates->elementSize(), indicesOuterSize);
            return false;
        }

        output->buffer().dimensions = dimension;

        auto shapeData = shape->host<int>();
        for (int i = 0; i < dimension; ++i) {
            output->setLength(i, shapeData[i]);
        }
        output->buffer().type = updates->buffer().type;

        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(updates)->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE_INPUTS(ShapeScatterNd, OpType_ScatterNd, (std::vector<int>{2}));
} // namespace MNN
