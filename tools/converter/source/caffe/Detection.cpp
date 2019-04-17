//
//  Detection.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class DetectionOutput : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    DetectionOutput() {
    }
    virtual ~DetectionOutput() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_DetectionOutput;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_DetectionOutput;
    }
};

void DetectionOutput::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters,
                          const caffe::LayerParameter& weight) {
    auto detectionOutputT             = new MNN::DetectionOutputT;
    dstOp->main.value                 = detectionOutputT;
    auto& caffeDetect                 = parameters.detection_output_param();
    detectionOutputT->backgroundLable = caffeDetect.background_label_id();

    detectionOutputT->classCount            = caffeDetect.num_classes();
    detectionOutputT->codeType              = caffeDetect.code_type();
    detectionOutputT->confidenceThreshold   = caffeDetect.confidence_threshold();
    detectionOutputT->keepTopK              = caffeDetect.keep_top_k();
    detectionOutputT->nmsThresholdold       = caffeDetect.nms_param().nms_threshold();
    detectionOutputT->nmsTopK               = caffeDetect.nms_param().top_k();
    detectionOutputT->shareLocation         = caffeDetect.share_location();
    detectionOutputT->varianceEncodedTarget = caffeDetect.variance_encoded_in_target();
    if (caffeDetect.has_objectness_score()) {
        detectionOutputT->objectnessScore = caffeDetect.objectness_score();
    }
}

static OpConverterRegister<DetectionOutput> a("DetectionOutput");
