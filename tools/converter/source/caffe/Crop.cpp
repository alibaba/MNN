//
//  Crop.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class Crop : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);

    Crop() {
    }
    virtual ~Crop() {
    }

    virtual MNN::OpType opType() {
        return MNN::OpType_Crop;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Crop;
    }
};

void Crop::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    auto cropParam = new MNN::CropT;

    auto& caffeCrop = parameters.crop_param();

    if (caffeCrop.has_axis()) {
        cropParam->axis = caffeCrop.axis();
    } else {
        cropParam->axis = 2;
    }

    cropParam->offset.resize(caffeCrop.offset_size());
    for (int i = 0; i < caffeCrop.offset_size(); ++i) {
        cropParam->offset[i] = caffeCrop.offset().data()[i];
    }

    dstOp->main.value = cropParam;
}

static OpConverterRegister<Crop> c("Crop");
