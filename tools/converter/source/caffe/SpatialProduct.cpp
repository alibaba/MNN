//
//  SpatialProduct.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class SpatialProduct : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    SpatialProduct() {
    }
    virtual ~SpatialProduct() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_SpatialProduct;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_NONE;
    }
};

void SpatialProduct::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters,
                         const caffe::LayerParameter& weight) {
}
static OpConverterRegister<SpatialProduct> a("SpatialProduct");
