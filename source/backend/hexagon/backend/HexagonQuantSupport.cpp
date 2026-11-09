#include "HexagonQuantization.hpp"

#include "MNN_generated.h"
#include "core/ConvolutionCommon.hpp"
#include "core/TensorUtils.hpp"

#include <cmath>

namespace MNN {

bool supportsW8A8Conv1x1(const Op* op, const std::vector<Tensor*>& inputs,
                         const std::vector<Tensor*>& outputs) {
    if (op == nullptr || op->type() != OpType_Convolution ||
        inputs.size() != 1 || outputs.size() != 1 ||
        inputs[0] == nullptr || outputs[0] == nullptr) {
        return false;
    }
    const auto conv = op->main_as_Convolution2D();
    if (conv == nullptr || conv->common() == nullptr ||
        conv->quanParameter() == nullptr) {
        return false;
    }
    const auto common = conv->common();
    const auto quant = conv->quanParameter();
    const bool quantStorageSupported =
        quant->type() == 1 ||
        (quant->type() == 4 &&
         !(quant->aMin() == 0 &&
           std::abs(quant->quantScale()) < 1e-6f));
    if (common->kernelX() != 1 || common->kernelY() != 1 ||
        common->strideX() != 1 || common->strideY() != 1 ||
        common->dilateX() != 1 || common->dilateY() != 1 ||
        common->group() != 1 || !quantStorageSupported ||
        quant->aMaxOrBits() != 8 || quant->readType() != 0 ||
        quant->index() != nullptr) {
        return false;
    }
    if (inputs[0]->dimensions() != 4 || outputs[0]->dimensions() != 4 ||
        TensorUtils::getDescribe(inputs[0])->dimensionFormat !=
            MNN_DATA_FORMAT_NC4HW4 ||
        TensorUtils::getDescribe(outputs[0])->dimensionFormat !=
            MNN_DATA_FORMAT_NC4HW4 ||
        inputs[0]->length(0) != outputs[0]->length(0) ||
        inputs[0]->length(2) != outputs[0]->length(2) ||
        inputs[0]->length(3) != outputs[0]->length(3) ||
        inputs[0]->length(1) <= 0 || outputs[0]->length(1) <= 0 ||
        (common->inputCount() > 0 &&
         common->inputCount() != inputs[0]->length(1)) ||
        (common->outputCount() > 0 &&
         common->outputCount() != outputs[0]->length(1))) {
        return false;
    }
    const auto pads =
        ConvolutionCommon::convolutionPad(inputs[0], outputs[0], common);
    if (pads.first != 0 || pads.second != 0) {
        return false;
    }

    const auto validTensorQuant = [](const Tensor* tensor) {
        const auto describe = TensorUtils::getDescribe(tensor);
        if (describe == nullptr || describe->quantAttr == nullptr) {
            return false;
        }
        const auto& attr = *describe->quantAttr;
        return attr.type == DataType_DT_INT8 &&
               std::isfinite(attr.scale) && attr.scale > 0.0f &&
               attr.zero == 0.0f && attr.min >= -127.0f &&
               attr.max <= 127.0f && attr.min <= attr.max;
    };
    return validTensorQuant(inputs[0]) && validTensorQuant(outputs[0]);
}

} // namespace MNN
