//
//  ShapeGatherND.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

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
        int batchDim = 0;
        if (nullptr != op->main_as_Axis()) {
            batchDim = op->main_as_Axis()->axis();
        }
        if (indices->elementSize() == 0) {
            outputs[0]->buffer().type = params->buffer().type;
            TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
            outputs[0]->buffer().dimensions = 2;
            outputs[0]->setLength(0, 0);
            outputs[0]->setLength(1, params->shape().back());
            return true;
        }
        auto indiceNd = indices->length(indices->dimensions()-1);
        if (indiceNd >  params->dimensions()) {
            MNN_ERROR("indiceNd >  params->dimensions()\n");
            return false;
        }
        outputs[0]->buffer().type = params->buffer().type;
        outputs[0]->buffer().dimensions = params->dimensions() + indices->dimensions() - indiceNd - 1 - batchDim;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        int outputIndex = 0;
        for (int i=0; i<indices->dimensions()-1; ++i) {
            outputs[0]->setLength(outputIndex++, indices->length(i));
        }
        for (int i=indiceNd + batchDim; i<params->dimensions(); ++i) {
            outputs[0]->setLength(outputIndex++, params->length(i));
        }
        return true;
    }
};

class GatherElementsComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        outputs[0]->buffer().dimensions = inputs[1]->dimensions();
        for (int i = 0; i < inputs[1]->dimensions(); i++) {
            outputs[0]->setLength(i, inputs[1]->length(i));
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        outputs[0]->buffer().type = inputs[0]->buffer().type;
        return true;
    }
};

REGISTER_SHAPE(GatherNDComputer, OpType_GatherND);
REGISTER_SHAPE(GatherElementsComputer, OpType_GatherElements);
} // namespace MNN
