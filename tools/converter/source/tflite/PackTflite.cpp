//
//  PackTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(PackTflite);
MNN::OpType PackTflite::opType(int quantizedModel) {
    return MNN::OpType_Pack;
}
MNN::OpParameter PackTflite::type(int quantizedModel) {
    return MNN::OpParameter_PackParam;
}

void PackTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
  auto param = new MNN::PackParamT;
  auto opt=tfliteOp->builtin_options.AsPackOptions();
  param->axis=opt->axis;
  dstOp->main.value = param;
}

DECLARE_OP_COVERTER(SplitTflite);
MNN::OpType SplitTflite::opType(int quantizedModel) {
    return MNN::OpType_Slice;
}
MNN::OpParameter SplitTflite::type(int quantizedModel) {
    return MNN::OpParameter_Slice;
}

void SplitTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
    auto param = new MNN::SliceT;
    param->sourceType = MNN::NetSource_TENSORFLOW;
    auto opt=tfliteOp->builtin_options.AsSplitOptions();
    param->slicePoints.resize(1);
    param->slicePoints[0] = opt->num_splits;
    dstOp->main.value = param;
    auto originInput = dstOp->inputIndexes[1];
    auto axisInput = dstOp->inputIndexes[0];
    const auto& axisTensor = tfliteTensors[axisInput];
    auto axisBuffer = tfliteModelBuffer[axisTensor->buffer].get();
    param->axis = ((int32_t*)(axisBuffer->data.data()))[0];

    dstOp->inputIndexes.resize(1);
    dstOp->inputIndexes[0] = originInput;
}

DECLARE_OP_COVERTER(StridedSliceTflite);
MNN::OpType StridedSliceTflite::opType(int quantizedModel) {
    return MNN::OpType_StridedSlice;
}
MNN::OpParameter StridedSliceTflite::type(int quantizedModel) {
    return MNN::OpParameter_StridedSliceParam;
}

void StridedSliceTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
    auto param = new MNN::StridedSliceParamT;
    param->T = MNN::DataType_DT_FLOAT;
    auto opt=tfliteOp->builtin_options.AsStridedSliceOptions();
    param->beginMask = opt->begin_mask;
    param->endMask = opt->end_mask;
    param->ellipsisMask = opt->ellipsis_mask;
    param->newAxisMask = opt->new_axis_mask;
    param->shrinkAxisMask = opt->shrink_axis_mask;
    dstOp->main.value = param;
}

using namespace tflite;
REGISTER_CONVERTER(PackTflite, BuiltinOperator_PACK);
REGISTER_CONVERTER(SplitTflite, BuiltinOperator_SPLIT);
REGISTER_CONVERTER(StridedSliceTflite, BuiltinOperator_STRIDED_SLICE);
