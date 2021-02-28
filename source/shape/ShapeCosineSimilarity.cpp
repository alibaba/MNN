//
//  ShapeCosineSimilarity.cpp
//  MNN
//
//  Created by MNN on 2019/7/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
class CosineSimilaritySize : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(3 <= inputs.size());
        auto x1        = inputs[0];
        auto x2        = inputs[1];
        auto dimTensor = inputs[2];
        const auto dim = dimTensor->host<int32_t>()[0];
        MNN_ASSERT(dim == 1);

        const int dimensions0 = x1->dimensions();
        const int dimensions1 = x2->dimensions();
        MNN_ASSERT(dimensions0 == dimensions1);
        for (int i = 0; i < dimensions0; ++i) {
            MNN_ASSERT(x1->length(i) == x2->length(i));
        }

        auto output                 = outputs[0];
        output->buffer().dimensions = dimensions0 - 1;
        for (int i = 0; i < dimensions0; ++i) {
            int index = i;
            if (i == dim) {
                continue;
            }
            if (i > dim) {
                index = i - 1;
            }
            output->setLength(index, x1->length(i));
        }
        output->buffer().type = x1->getType();
        TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        return true;
    }
};

REGISTER_SHAPE_INPUTS(CosineSimilaritySize, OpType_CosineSimilarity, (std::vector<int>{2}));

} // namespace MNN
