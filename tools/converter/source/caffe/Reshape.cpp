//
//  Reshape.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "logkit.h"

class Reshape : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    Reshape() {
    }
    virtual ~Reshape() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Reshape;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Reshape;
    }
};

void Reshape::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    auto reshape      = new MNN::ReshapeT;
    dstOp->main.value = reshape;
    auto c            = parameters.reshape_param();
    DCHECK(c.has_shape()) << "Reshape Param ERROR!";
    auto shape = c.shape();
    for (int i = 0; i < shape.dim_size(); ++i) {
        reshape->dims.push_back(shape.dim(i));
    }
}
static OpConverterRegister<Reshape> a("Reshape");

class Flatten : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    Flatten() {
    }
    virtual ~Flatten() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Reshape;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Reshape;
    }
};

void Flatten::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    const ::caffe::FlattenParameter& par = parameters.flatten_param();
    int axis = 1, endAxis = 4;
    if (par.has_axis()) {
        axis = par.axis();
    }
    if (par.has_end_axis()) {
        endAxis = par.end_axis();
    }

    auto reshape      = new MNN::ReshapeT;
    dstOp->main.value = reshape;
    for (int i = 0; i < axis; ++i) {
        reshape->dims.push_back(0);
    }
    reshape->dims.push_back(-1);
    for (int i = axis + 1; i < endAxis; ++i) {
        reshape->dims.push_back(1);
    }
    for (int i = endAxis; i < 4; ++i) {
        reshape->dims.push_back(0);
    }
}
static OpConverterRegister<Flatten> __a("Flatten");
