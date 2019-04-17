//
//  Slice.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
class Slice : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    Slice() {
    }
    virtual ~Slice() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Slice;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Slice;
    }
};

void Slice::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    auto slice        = new MNN::SliceT;
    dstOp->main.value = slice;
    auto c            = parameters.slice_param();
    slice->axis       = c.axis();
    for (int i = 0; i < c.slice_point_size(); ++i) {
        slice->slicePoints.push_back(c.slice_point(i));
    }
}
static OpConverterRegister<Slice> a("Slice");
