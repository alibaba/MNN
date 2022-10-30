//
//  CoreMLInterp.cpp
//  MNN
//
//  Created by MNN on 2021/05/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLInterp.hpp"

namespace MNN {

CoreMLInterp::CoreMLInterp(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
    initLayer();
}

ErrorCode CoreMLInterp::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    auto interpParam = mOp->main_as_Interp();
    // ResizeBilinear: NPU; UpsampleLayer: GPU ?
// #define USE_RESIZE_BILINEAR
#ifdef USE_RESIZE_BILINEAR
    mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_RESIZE_BILINEAR;
    mLayer_->resizebilinear = mCoreMLBackend->create<CoreML__Specification__ResizeBilinearLayerParams>();
    core_ml__specification__resize_bilinear_layer_params__init(mLayer_->resizebilinear);
    mLayer_->resizebilinear->n_targetsize = 2;
    mLayer_->resizebilinear->targetsize = mCoreMLBackend->create<uint64_t>(mLayer_->resizebilinear->n_targetsize);
    mLayer_->resizebilinear->targetsize[0] = inputs[0]->height() / interpParam->heightScale();
    mLayer_->resizebilinear->targetsize[1] = inputs[0]->width() / interpParam->widthScale();
    mLayer_->resizebilinear->mode = mCoreMLBackend->create<CoreML__Specification__SamplingMode>();
    core_ml__specification__sampling_mode__init(mLayer_->resizebilinear->mode);
    mLayer_->resizebilinear->mode->samplingmethod = CORE_ML__SPECIFICATION__SAMPLING_MODE__METHOD__UPSAMPLE_MODE;
#else
    mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_UPSAMPLE;
    mLayer_->upsample = mCoreMLBackend->create<CoreML__Specification__UpsampleLayerParams>();
    core_ml__specification__upsample_layer_params__init(mLayer_->upsample);
    float heightScale = 1.0 / interpParam->heightScale();
    float widthScale  = 1.0 / interpParam->widthScale();
    uint64_t heightScaleI = static_cast<uint64_t>(heightScale);
    uint64_t widthScaleI = static_cast<uint64_t>(widthScale);
    if (heightScale - heightScaleI == 0 && widthScale - widthScaleI == 0) {
        mLayer_->upsample->n_scalingfactor = 2;
        mLayer_->upsample->scalingfactor =
            mCoreMLBackend->create<uint64_t>(mLayer_->upsample->n_scalingfactor);
        mLayer_->upsample->scalingfactor[0] = heightScaleI;
        mLayer_->upsample->scalingfactor[1] = widthScaleI;
    } else {
        // scale
        mLayer_->upsample->n_fractionalscalingfactor = 2;
        mLayer_->upsample->fractionalscalingfactor =
            mCoreMLBackend->create<float>(mLayer_->upsample->n_fractionalscalingfactor);
        mLayer_->upsample->fractionalscalingfactor[0] = heightScale;
        mLayer_->upsample->fractionalscalingfactor[1] = widthScale;
    }
    // mode
    if (interpParam->resizeType() == 1 && mLayer_->upsample->n_fractionalscalingfactor != 2) {
        mLayer_->upsample->mode = CORE_ML__SPECIFICATION__UPSAMPLE_LAYER_PARAMS__INTERPOLATION_MODE__NN;
    } else if (interpParam->resizeType() == 2) {
        mLayer_->upsample->mode = CORE_ML__SPECIFICATION__UPSAMPLE_LAYER_PARAMS__INTERPOLATION_MODE__BILINEAR;
        // align corner
        mLayer_->upsample->linearupsamplemode = interpParam->alignCorners() ?
            CORE_ML__SPECIFICATION__UPSAMPLE_LAYER_PARAMS__LINEAR_UPSAMPLE_MODE__ALIGN_CORNERS_TRUE :
            CORE_ML__SPECIFICATION__UPSAMPLE_LAYER_PARAMS__LINEAR_UPSAMPLE_MODE__ALIGN_CORNERS_FALSE;
    } else {
        MNN_ERROR("[CoreML] Interp Don't support [Cubic, NearestneighborRound] mode.");
        return NOT_SUPPORT;
    }
#endif
    setLayerInputsAndOutputs(mLayer_, {mCoreMLBackend->getTensorName(inputs[0])},
                                      {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->addLayer(mLayer_);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLInterp, OpType_Interp)
} // namespace MNN
