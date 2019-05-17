//
//  Eltwise.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "logkit.h"

class EltWise : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    EltWise() {
    }
    virtual ~EltWise() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Eltwise;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Eltwise;
    }
};

void EltWise::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    auto elt          = new MNN::EltwiseT;
    dstOp->main.value = elt;
    auto& caffeParam  = parameters.eltwise_param();
    switch (caffeParam.operation()) {
        case caffe::EltwiseParameter_EltwiseOp_MAX:
            elt->type = MNN::EltwiseType_MAXIMUM;
            break;
        case caffe::EltwiseParameter_EltwiseOp_SUM:
            elt->type = MNN::EltwiseType_SUM;
            break;
        case caffe::EltwiseParameter_EltwiseOp_PROD:
            elt->type = MNN::EltwiseType_PROD;
            break;

        default:
            break;
    }

    const int coffSize = caffeParam.coeff_size();
    elt->coeff.resize(coffSize);
    for (int i = 0; i < coffSize; ++i) {
        elt->coeff[i] = caffeParam.coeff(i);
    }
}
static OpConverterRegister<EltWise> a("Eltwise");
