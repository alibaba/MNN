//
//  Concat.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Concat : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    Concat() {
    }
    virtual ~Concat() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Concat;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Axis;
    }
};

void Concat::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    auto axisT        = new MNN::AxisT;
    dstOp->main.value = axisT;
    auto& c           = parameters.concat_param();
    if (c.has_axis()) {
        axisT->axis = c.axis();
    } else {
        axisT->axis = 1;
    }
}
static OpConverterRegister<Concat> a("Concat");
