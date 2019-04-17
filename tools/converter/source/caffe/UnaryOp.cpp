//
//  UnaryOp.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class UnaryOp : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    UnaryOp() {
    }
    virtual ~UnaryOp() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_UnaryOp;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_UnaryOp;
    }
};

void UnaryOp::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    auto parameter = new MNN::UnaryOpT;

    parameter->T = MNN::DataType_DT_FLOAT;

    parameter->opType = MNN::UnaryOpOperation_ABS;

    dstOp->main.value = parameter;
}

static OpConverterRegister<UnaryOp> ____a("AbsVal");
