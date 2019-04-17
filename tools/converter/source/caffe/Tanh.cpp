//
//  Tanh.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Tanh : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    Tanh() {
    }
    virtual ~Tanh() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_TanH;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_NONE;
    }
};

void Tanh::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    dstOp->main.value = nullptr;
}
static OpConverterRegister<Tanh> a("TanH");
