//
//  Threshold.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Threshold : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    Threshold() {
    }
    virtual ~Threshold() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Threshold;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_ELU;
    }
};

void Threshold::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    auto threshold = parameters.threshold_param().threshold();
    auto parameter = new MNN::ELUT;
    parameter->alpha = threshold;
    dstOp->main.value = parameter;
}

static OpConverterRegister<Threshold> ____a("Threshold");
