//
// Created by alibaba on 2019/9/11.
//

#include "TRTSoftmax.hpp"
#include "TRTBackend.hpp"
#include <core/TensorUtils.hpp>

using namespace std;

namespace MNN {

TRTSoftmax::TRTSoftmax(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b,op) {
        int axis = mOp->main_as_Axis()->axis();
        mAxis = axis < 0 ? axis + outputs[0]->dimensions(): axis;
    }

std::vector<ITensor *> TRTSoftmax::onEncode(const std::vector<ITensor *> &xOp) {
    
    auto softmax_layer = mTrtBackend->getNetwork()->addSoftMax(*(xOp[0]));
    softmax_layer->setAxes(1U << mAxis);
    return {softmax_layer->getOutput(0)};
}

TRTCreatorRegister<TypedCreator<TRTSoftmax>> __softmax_op(OpType_Softmax);

} // namespace MNN
