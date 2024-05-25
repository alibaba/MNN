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
        //MNN_CHECK(indices->length(indicesDimension - 1) == 1, "indices.shape[-1] = shape.rank");

        const int outerDims = indicesDimension - 1;
        const int dimension = shape->length(0);
        //MNN_CHECK(updates->dimensions() == dimension, "updates dimension should be equal to given shape");

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
