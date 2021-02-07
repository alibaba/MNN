#include "TRTInterp.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"
#include "schema/current/MNNPlugin_generated.h"

using namespace std;

namespace MNN {

static float resizeScale(int inputSize, int outputSize, bool isAlign) {
    int corner = 0;
    if (isAlign) {
        corner = 1;
    }
    return (float)(inputSize - corner) / (float)(outputSize - corner);
}

TRTInterp::TRTInterp(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
    // Do nothing
}

std::vector<ITensor *> TRTInterp::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    MNN_PRINT("\n\nTRTInterp in\n\n");
#endif
    auto plu = createPluginWithOutput(mOutputs);

    int inputChannel = mInputs[0]->channel();
    int inputBatch   = mInputs[0]->batch();

    int inputHeight  = mInputs[0]->height();
    int inputWidth   = mInputs[0]->width();
    int outputHeight = mOutputs[0]->height();
    int outputWidth  = mOutputs[0]->width();

    bool alignCorners = mOp->main_as_Interp()->alignCorners();
    // TODO, not used now
    bool halfPixelCenters = mOp->main_as_Interp()->halfPixelCenters();
    int resizeType        = mOp->main_as_Interp()->resizeType();
    if(resizeType != 1 && resizeType != 2) {
        MNN_PRINT("Interp Type not support!\n");
    }
    plu->main.type  = MNNTRTPlugin::Parameter_InterpInfo;
    plu->main.value = new MNNTRTPlugin::InterpInfoT;
    auto interp     = plu->main.AsInterpInfo();

    interp->inputChannel  = inputChannel;
    interp->heightScale   = resizeScale(inputHeight, outputHeight, alignCorners);
    interp->widthScale    = resizeScale(inputWidth, outputWidth, alignCorners);
    interp->channelBlocks = inputChannel * inputBatch;
    interp->outputWidth   = outputWidth;
    interp->outputH_N     = outputHeight * inputBatch;
    interp->inputHeight   = inputHeight;
    interp->inputWidth    = inputWidth;
    interp->outputHeight  = outputHeight;
    // MNN_PRINT("hs:%f, ws:%f, c:%d, h:%d, w:%d\n", interp->heightScale, interp->widthScale, interp->channelBlocks,
    // interp->outputHeight, interp->outputWidth);

    auto interpPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
    nvinfer1::IPluginLayer *plugin =
        mTrtBackend->getNetwork()->addPluginExt(&xOp[0], 1, *((nvinfer1::IPluginExt *)interpPlugin));
    if (plugin == nullptr) {
        MNN_PRINT("Interp plugin == nullptr !!!\n");
    }
    mTrtBackend->pushReleaseLayer(interpPlugin);
    return {plugin->getOutput(0)};
}

TRTCreatorRegister<TypedCreator<TRTInterp>> __interp_op(OpType_Interp);

}