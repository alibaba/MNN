//
//  ShapeMoments.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNDefine.h"
#include "SizeComputer.hpp"

namespace MNN {
class MomentsComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(2 == outputs.size());

        auto input        = inputs[0];
        auto mean         = outputs[0];
        auto variance     = outputs[1];
        auto momentsParam = op->main_as_MomentsParam();
        mean->setType(momentsParam->dType());
        variance->setType(momentsParam->dType());
        if (nullptr == momentsParam->dim()) {
            mean->buffer().dimensions     = 0;
            variance->buffer().dimensions = 0;
            return true;
        }

        std::set<int> momentsDims;
        for (int i = 0; i < momentsParam->dim()->size(); ++i) {
            momentsDims.insert(momentsParam->dim()->data()[i]);
        }
        std::vector<int> outputShape;
        for (int i = 0; i < input->dimensions(); ++i) {
            if (momentsDims.find(i) == momentsDims.end()) {
                outputShape.push_back(input->length(i));
            } else if (momentsParam->keepDims()) {
                outputShape.push_back(1);
            }
        }

        const auto outputDim          = outputShape.size();
        mean->buffer().dimensions     = static_cast<int>(outputDim);
        variance->buffer().dimensions = static_cast<int>(outputDim);
        for (int i = 0; i < outputDim; ++i) {
            mean->setLength(i, outputShape[i]);
            variance->setLength(i, outputShape[i]);
        }

        return true;
    }
};

REGISTER_SHAPE(MomentsComputer, OpType_Moments);

} // namespace MNN
