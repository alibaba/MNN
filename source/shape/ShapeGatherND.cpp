//
//  ShapeGatherND.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "core/SizeComputer.hpp"

namespace MNN {
class GatherNDComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto params  = inputs[0];
        auto indices = inputs[1];
        if(indices->getType().code != halide_type_int) {
            MNN_ERROR("Don't support not int indices\n");
            return false;
        }
        if (params->dimensions() < 1 || indices->dimensions() < 1) {
            MNN_ERROR("params->dimensions() < 1 || indices->dimensions() < 1\n");
            return false;
        }
        auto indiceNd = indices->length(indices->dimensions()-1);
        if (indiceNd >  params->dimensions()) {
            MNN_ERROR("indiceNd >  params->dimensions()\n");
            return false;
        }
        outputs[0]->buffer().type = params->buffer().type;
        outputs[0]->buffer().dimensions = params->dimensions() + indices->dimensions() - indiceNd -1;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        int outputIndex = 0;
        for (int i=0; i<indices->dimensions()-1; ++i) {
            outputs[0]->setLength(outputIndex++, indices->length(i));
        }
        for (int i=indiceNd; i<params->dimensions(); ++i) {
            outputs[0]->setLength(outputIndex++, params->length(i));
        }
        return true;
    }
};

REGISTER_SHAPE(GatherNDComputer, OpType_GatherND);
} // namespace MNN
