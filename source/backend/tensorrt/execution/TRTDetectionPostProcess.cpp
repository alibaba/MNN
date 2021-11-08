//
//  TRTBinary.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTDetectionPostProcess.hpp"
#include <core/TensorUtils.hpp>
#include "schema/current/MNNPlugin_generated.h"
using namespace std;
namespace MNN {

TRTDetectionPostProcess::TRTDetectionPostProcess(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
    auto param = op->main_as_DetectionPostProcessParam();
    param->UnPackTo(&mParam);
    if (mParam.useRegularNMS) {
        MNN_ERROR("TODO, use regular NMS to process decoded boxes!");
        return;
    }
}

std::vector<ITensor *> TRTDetectionPostProcess::onEncode(const std::vector<ITensor *> &xOp) {
    auto plu = createPluginWithOutput(mOutputs);
    plu->main.type  = MNNTRTPlugin::Parameter_DetectionPostProcessInfo;
    plu->main.value = new MNNTRTPlugin::DetectionPostProcessInfoT;
    auto detection     = plu->main.AsDetectionPostProcessInfo();

    auto boxesEncoding = mInputs[0];
    auto classPredictions = mInputs[1];
    auto anchors = mInputs[2];

    detection->numAnchors0 = mInputs[0]->length(1);
    detection->scaleValues.push_back(mParam.centerSizeEncoding[0]);
    detection->scaleValues.push_back(mParam.centerSizeEncoding[1]);
    detection->scaleValues.push_back(mParam.centerSizeEncoding[2]);
    detection->scaleValues.push_back(mParam.centerSizeEncoding[3]);

    detection->numBoxes = boxesEncoding->length(1);
    detection->boxCoordNum = boxesEncoding->length(2);
    detection->numAnchors1 = anchors->length(0);
    detection->anchorsCoordNum = anchors->length(1);

    detection->numClassWithBackground = classPredictions->length(2);

    detection->numClasses = mParam.numClasses;
    detection->maxClassesPerAnchor = mParam.maxClassesPerDetection;

    detection->maxDetections = mParam.maxDetections;
    detection->iouThreshold = mParam.iouThreshold;
    detection->nmsScoreThreshold = mParam.nmsScoreThreshold;

    auto detectionPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
    nvinfer1::IPluginLayer *plugin = mTrtBackend->getNetwork()->addPluginExt(&xOp[0], 3, *((nvinfer1::IPluginExt *)detectionPlugin));
    if (plugin == nullptr) {
        printf("Interp plugin == nullptr !!!\n");
    }
    mTrtBackend->pushReleaseLayer(detectionPlugin);
    return {plugin->getOutput(0), plugin->getOutput(1), plugin->getOutput(2), plugin->getOutput(3)};
}

TRTCreatorRegister<TypedCreator<TRTDetectionPostProcess>> __detection_post_process_op(OpType_DetectionPostProcess);


} // namespace MNN
