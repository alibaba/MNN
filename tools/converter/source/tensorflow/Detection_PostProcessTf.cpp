//
//  Detection_PostProcessTf.cpp
//  MNNConverter
//
//  Created by MNN on b'2019/11/21'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

//
//  Detection_PostProcessTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(Detection_PostProcessTf);

MNN::OpType Detection_PostProcessTf::opType(){
    return MNN::OpType_DetectionPostProcess;
}

MNN::OpParameter Detection_PostProcessTf::type(){
    return MNN::OpParameter_DetectionPostProcessParam;
}

void Detection_PostProcessTf::run(MNN::OpT *dstOp, TmpNode *srcNode){
    auto postProcessParam = new MNN::DetectionPostProcessParamT;
    tensorflow::AttrValue value;
    if(find_attr_value(srcNode->tfNode, "max_detections", value)){
        postProcessParam->maxDetections = value.i();
    }
    if(find_attr_value(srcNode->tfNode, "max_classes_per_detection", value)){
        postProcessParam->maxClassesPerDetection = value.i();
    }
    if(find_attr_value(srcNode->tfNode, "detections_per_class", value)){
        postProcessParam->detectionsPerClass = value.i();
    }
    if(find_attr_value(srcNode->tfNode, "use_regular_nms", value)){
        postProcessParam->useRegularNMS = value.b();
    }
    if(find_attr_value(srcNode->tfNode, "nms_score_threshold", value)){
        postProcessParam->nmsScoreThreshold = value.f();
    }
    if(find_attr_value(srcNode->tfNode, "nms_iou_threshold", value)){
        postProcessParam->iouThreshold = value.f();
    }
    if(find_attr_value(srcNode->tfNode, "num_classes", value)){
        postProcessParam->numClasses = value.i();
    }
    if(find_attr_value(srcNode->tfNode, "y_scale", value)){
        postProcessParam->centerSizeEncoding.push_back(value.f());
    }
    if(find_attr_value(srcNode->tfNode, "x_scale", value)){
        postProcessParam->centerSizeEncoding.push_back(value.f());
    }
    if(find_attr_value(srcNode->tfNode, "h_scale", value)){
        postProcessParam->centerSizeEncoding.push_back(value.f());
    }
    if(find_attr_value(srcNode->tfNode, "w_scale", value)){
        postProcessParam->centerSizeEncoding.push_back(value.f());
    }
    dstOp->main.value = postProcessParam;
    
    // Detection_PostProcessTf output 4 tensors
    dstOp->outputIndexes = {-1, -1, -1, -1};
}

REGISTER_CONVERTER(Detection_PostProcessTf, TFLite_Detection_PostProcess);
