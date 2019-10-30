//
//  PriorBox.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class PrioxBox : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    PrioxBox() {
    }
    virtual ~PrioxBox() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_PriorBox;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_PriorBox;
    }
};

void PrioxBox::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    auto prior = new MNN::PriorBoxT;

    dstOp->main.value = prior;
    auto& caffePrior  = parameters.prior_box_param();
    for (int i = 0; i < caffePrior.aspect_ratio_size(); ++i) {
        prior->aspectRatios.push_back(caffePrior.aspect_ratio(i));
    }
    for (int i = 0; i < caffePrior.min_size_size(); ++i) {
        prior->minSizes.push_back(caffePrior.min_size(i));
    }
    for (int i = 0; i < caffePrior.max_size_size(); ++i) {
        prior->maxSizes.push_back(caffePrior.max_size(i));
    }
    for (int i = 0; i < caffePrior.variance_size(); ++i) {
        prior->variances.push_back(caffePrior.variance(i));
    }
    prior->clip        = caffePrior.clip();
    prior->flip        = caffePrior.flip();
    prior->imageWidth  = 0;
    prior->imageHeight = 0;
    if (caffePrior.has_img_w()) {
        prior->imageWidth = caffePrior.img_w();
    }
    if (caffePrior.has_img_size()) {
        prior->imageWidth  = caffePrior.img_size();
        prior->imageHeight = caffePrior.img_size();
    }
    if (caffePrior.has_img_h()) {
        prior->imageHeight = caffePrior.img_h();
    }
    prior->offset = 0.5f;
    if (caffePrior.has_offset()) {
        prior->offset = caffePrior.offset();
    }
    if (caffePrior.has_step()) {
        prior->stepWidth  = caffePrior.step();
        prior->stepHeight = caffePrior.step();
    } else if (caffePrior.has_step_h() && caffePrior.has_step_w()) {
        prior->stepWidth  = caffePrior.step_w();
        prior->stepHeight = caffePrior.step_h();
    } else {
        prior->stepWidth  = 0;
        prior->stepHeight = 0;
    }
}
static OpConverterRegister<PrioxBox> a("PriorBox");
