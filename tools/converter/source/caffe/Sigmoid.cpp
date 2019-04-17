//
//  Sigmoid.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Sigmoid : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    Sigmoid() {
    }
    virtual ~Sigmoid() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Sigmoid;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_NONE;
    }
};

void Sigmoid::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    dstOp->main.value = nullptr;
}

static OpConverterRegister<Sigmoid> a("Sigmoid");
