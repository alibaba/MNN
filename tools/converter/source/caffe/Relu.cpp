//
//  Relu.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "logkit.h"

class Relu : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    Relu() {
    }
    virtual ~Relu() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_ReLU;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Relu;
    }
};

class Relu6 : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    Relu6() {
    }
    virtual ~Relu6() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_ReLU6;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Relu6;
    }
};

void Relu::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    auto relu = new MNN::ReluT;
    if (parameters.relu_param().has_negative_slope()) {
        relu->slope = parameters.relu_param().negative_slope();
    } else {
        relu->slope = 0.0f;
    }
    dstOp->main.value = relu;
}

static OpConverterRegister<Relu> a("ReLU");

void Relu6::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    auto relu6        = new MNN::Relu6T;
    dstOp->main.value = relu6;
}

static OpConverterRegister<Relu6> b("ReLU6");

class PRelu : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
        auto relu = new MNN::PReluT;
        auto v0w  = &weight;
        DCHECK(v0w->blobs_size() >= 1) << "caffemodel error!";
        const caffe::BlobProto& slope_blob = v0w->blobs(0);
        relu->slopeCount                   = slope_blob.data_size();
        relu->slope.resize(relu->slopeCount);

        memcpy(relu->slope.data(), slope_blob.data().data(), sizeof(float) * relu->slopeCount);
        dstOp->main.value = relu;
    }
    PRelu() {
    }
    virtual ~PRelu() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_PReLU;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_PRelu;
    }
};

static OpConverterRegister<PRelu> __a("PReLU");
