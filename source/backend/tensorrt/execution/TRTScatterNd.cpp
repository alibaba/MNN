#include "TRTScatterNd.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"
#include "schema/current/MNNPlugin_generated.h"

using namespace std;

namespace MNN {

TRTScatterNd::TRTScatterNd(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
    // Do nothing
}

std::vector<ITensor *> TRTScatterNd::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    MNN_PRINT("\n\nTRTScatterNd in\n\n");
#endif
    auto plu        = createPluginWithOutput(mOutputs);
    plu->main.type  = MNNTRTPlugin::Parameter_ScatterNdInfo;
    plu->main.value = new MNNTRTPlugin::ScatterNdInfoT;
    auto scatter    = plu->main.AsScatterNdInfo();

    MNN_ASSERT(mInputs.size() == 3);
    auto indices               = mInputs[0];
    auto updates               = mInputs[1];
    auto shape                 = mInputs[2];
    auto output                = mOutputs[0];
    const int indicesDimension = indices->dimensions();
    scatter->indicesLastDim    = indices->length(indicesDimension - 1);
    scatter->indexes           = indices->elementSize() / scatter->indicesLastDim;

    scatter->accNumber = 1;
    for (int i = indicesDimension - 1; i < updates->dimensions(); ++i) {
        scatter->accNumber *= updates->length(i);
    }

    const int outputElementSize = output->elementSize();
    scatter->outElementSize     = outputElementSize;
    int remainSize              = outputElementSize;
    std::vector<int> temp(scatter->indicesLastDim, 0);
    for (int i = 0; i < scatter->indicesLastDim; ++i) {
        temp[i]    = remainSize / output->length(i);
        remainSize = temp[i];
    }
    scatter->dimsToCount.assign(temp.begin(), temp.end());

    auto scatterNdPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
    nvinfer1::IPluginLayer *plugin =
        mTrtBackend->getNetwork()->addPluginExt(&xOp[0], 2, *((nvinfer1::IPluginExt *)scatterNdPlugin));
    if (plugin == nullptr) {
        MNN_PRINT("scatterNd plugin == nullptr !!!\n");
    }
    mTrtBackend->pushReleaseLayer(scatterNdPlugin);
    return {plugin->getOutput(0)};
}

TRTCreatorRegister<TypedCreator<TRTScatterNd>> __scatterNd_op(OpType_ScatterNd);

}