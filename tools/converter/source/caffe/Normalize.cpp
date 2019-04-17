//
//  Normalize.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Normalize : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    Normalize() {
    }
    virtual ~Normalize() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Normalize;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Normalize;
    }
};

void Normalize::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    auto normizeT           = new MNN::NormalizeT;
    dstOp->main.value       = normizeT;
    auto& l                 = parameters.norm_param();
    normizeT->channelShared = l.channel_shared();
    normizeT->acrossSpatial = l.across_spatial();
    normizeT->eps           = l.eps();
    auto& scaleBlob         = weight.blobs(0);
    for (int i = 0; i < scaleBlob.data_size(); ++i) {
        normizeT->scale.push_back(scaleBlob.data(i));
    }
}
static OpConverterRegister<Normalize> a("Normalize");
