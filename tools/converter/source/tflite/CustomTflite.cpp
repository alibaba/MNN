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
struct TfLiteTransposeConvParams{
  // Parameters supported by version 1:
  int padding = 0;
  int stride_width;
  int stride_height;

  // Parameters supported by version 4:
  int activation = 0;

  // Parameters for TransposeConv version 5 or above.
  // Used to determine the default value for the quantized bias.
  int quantized_bias_type = 0;
};


void CustomTflite::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet, int quantizedModel) {
    auto &customOPCode = tfliteOpSet[tfliteOp->opcode_index]->custom_code;
    if (customOPCode == "Convolution2DTransposeBias") {
        dstOp->type = MNN::OpType_Deconvolution;
        TfLiteTransposeConvParams params;
        size_t copyLenth = std::min(sizeof(params), tfliteOp->custom_options.size());
        ::memcpy(&params, tfliteOp->custom_options.data(), copyLenth);
        dstOp->main.type = MNN::OpParameter_Convolution2D;
        dstOp->main.value = new MNN::Convolution2DT;
        auto conv = dstOp->main.AsConvolution2D();
        conv->common.reset(new MNN::Convolution2DCommonT);
        auto common = conv->common.get();
        common->strideX = params.stride_width;
        common->strideY = params.stride_height;
        switch (params.padding) {
            case 0:
                common->padMode = MNN::PadMode_CAFFE;
                break;
            case 1:
                common->padMode = MNN::PadMode_SAME;
                break;
            case 2:
                common->padMode = MNN::PadMode_VALID;
                break;
            default:
                break;
        }
        const int inputIndex     = tfliteOp->inputs[0];
        const int weightIndex    = tfliteOp->inputs[1];
        const int biasIndex    = tfliteOp->inputs[2];
        const int outputIndex    = tfliteOp->outputs[0];
        const auto& inputTensor  = tfliteTensors[inputIndex];
        const auto& weightTensor = tfliteTensors[weightIndex];
        const auto& biasTensor = tfliteTensors[biasIndex];
        
        const auto& weightShape = weightTensor->shape;
        DCHECK(weightShape.size() == 4) << "Conv2D weight ERROR!";
        const int co         = weightShape[0];
        const int kh         = weightShape[1];
        const int kw         = weightShape[2];
        const int ci         = weightShape[3];
        
        // TODO: Support group
        common->group = 1;
        common->outputCount = co;
        common->inputCount = ci;
        common->kernelX = kw;
        common->kernelY = kh;
        
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(MNN::Op::Pack(builder, dstOp));
        dstOp->type = MNN::OpType_Extra;
        dstOp->main.Reset();
        dstOp->main.value = new MNN::ExtraT;
        dstOp->main.type = MNN::OpParameter_Extra;
        auto extra = dstOp->main.AsExtra();
        extra->type = "Convolution2DTransposeBias";
        extra->engine = "Tflite";
        extra->info.resize(builder.GetSize());
        ::memcpy(extra->info.data(), builder.GetBufferPointer(), builder.GetSize());
        return;
    }
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
