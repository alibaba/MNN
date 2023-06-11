//
//  CustomTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfliteUtils.hpp"
#include "flatbuffers/flexbuffers.h"
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(CustomTflite);

MNN::OpType CustomTflite::opType(int quantizedModel) {
    DCHECK(!quantizedModel) << "Not support quantized model";
    return MNN::OpType_DetectionPostProcess;
}

MNN::OpParameter CustomTflite::type(int quantizedModel) {
    return MNN::OpParameter_DetectionPostProcessParam;
}

void CustomTflite::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet, int quantizedModel) {
    auto &customOPCode = tfliteOpSet[tfliteOp->opcode_index]->custom_code;
    DCHECK(customOPCode == "TFLite_Detection_PostProcess")
        << "Now Only support Custom op of 'TFLite_Detection_PostProcess'";

    auto postProcessParam    = new MNN::DetectionPostProcessParamT;
    auto customOptionsFormat = tfliteOp->custom_options_format;
    DCHECK(customOptionsFormat == tflite::CustomOptionsFormat_FLEXBUFFERS) << "custom options format ERROR!";
    const uint8_t *customOptionBufferDataPtr = tfliteOp->custom_options.data();
    const auto size                          = tfliteOp->custom_options.size();
    const flexbuffers::Map &m                = flexbuffers::GetRoot(customOptionBufferDataPtr, size).AsMap();

    postProcessParam->maxDetections          = m["max_detections"].AsInt32();
    postProcessParam->maxClassesPerDetection = m["max_classes_per_detection"].AsInt32();
    if (m["detections_per_class"].IsNull()) {
        postProcessParam->detectionsPerClass = 100;
    } else {
        postProcessParam->detectionsPerClass = m["detections_per_class"].AsInt32();
    }
    if (m["use_regular_nms"].IsNull()) {
        postProcessParam->useRegularNMS = false;
    } else {
        postProcessParam->useRegularNMS = m["use_regular_nms"].AsBool();
    }
    postProcessParam->nmsScoreThreshold = m["nms_score_threshold"].AsFloat();
    postProcessParam->iouThreshold      = m["nms_iou_threshold"].AsFloat();
    postProcessParam->numClasses        = m["num_classes"].AsInt32();
    postProcessParam->centerSizeEncoding.push_back(m["y_scale"].AsFloat());
    postProcessParam->centerSizeEncoding.push_back(m["x_scale"].AsFloat());
    postProcessParam->centerSizeEncoding.push_back(m["h_scale"].AsFloat());
    postProcessParam->centerSizeEncoding.push_back(m["w_scale"].AsFloat());

    dstOp->main.value = postProcessParam;

    DCHECK(tfliteOp->inputs.size() == 3) << "TFLite_Detection_PostProcess should have 3 inputs!";
    DCHECK(tfliteOp->outputs.size() == 4) << "TFLite_Detection_PostProcess should have 4 outputs!";
}

using namespace tflite;
REGISTER_CONVERTER(CustomTflite, BuiltinOperator_CUSTOM);
