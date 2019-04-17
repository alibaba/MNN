//
//  LRN.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Lrn : public OpConverter {
public:
    void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    virtual MNN::OpType opType() {
        return MNN::OpType_LRN;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_LRN;
    }
};

void Lrn::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    MNN::LRNT* lrn    = new MNN::LRNT;
    dstOp->main.value = lrn;

    auto caffeLrn   = parameters.lrn_param();
    lrn->alpha      = caffeLrn.alpha();
    lrn->beta       = caffeLrn.beta();
    lrn->localSize  = caffeLrn.local_size();
    lrn->regionType = caffeLrn.norm_region();
}

static OpConverterRegister<Lrn> a("LRN");
static OpConverterRegister<Lrn> _a("CuDNNLRNCrossChannel");
