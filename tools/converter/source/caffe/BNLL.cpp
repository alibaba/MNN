//
//  BNLL.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class BNLL : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    BNLL() {
    }
    virtual ~BNLL() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_UnaryOp;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_UnaryOp;
    }
};

void BNLL::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    auto parameter = new MNN::UnaryOpT;

    parameter->T = MNN::DataType_DT_FLOAT;

    parameter->opType = MNN::UnaryOpOperation_BNLL;

    dstOp->main.value = parameter;
}

static OpConverterRegister<BNLL> ____a("BNLL");
