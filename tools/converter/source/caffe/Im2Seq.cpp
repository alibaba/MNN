//
//  Im2Seq.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Im2Seq : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    Im2Seq() {
    }
    virtual ~Im2Seq() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Im2Seq;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_NONE;
    }
};

void Im2Seq::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
}
static OpConverterRegister<Im2Seq> a("Im2seq");

class Seq2Out : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    }
    Seq2Out() {
    }
    virtual ~Seq2Out() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Seq2Out;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_NONE;
    }
};
static OpConverterRegister<Seq2Out> b("Seq2out");
