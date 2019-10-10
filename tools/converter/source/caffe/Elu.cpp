//
//  Elu.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "logkit.h"

class Elu : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    Elu() {
    }
    virtual ~Elu() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_ELU;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_ELU;
    }
};

void Elu::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    auto elu = new MNN::ELUT;
    auto param = parameters.elu_param();
    elu->alpha = param.alpha();
    dstOp->main.value = elu;
}

static OpConverterRegister<Elu> a("ELU");
