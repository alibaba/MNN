//
//  ShapePriorbox.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
class PriorBoxComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto layer = op->main_as_PriorBox();

        auto inputTensor  = inputs[0];
        auto inputTensor1 = inputs[1];

        int w = inputTensor->width();
        int h = inputTensor->height();

        auto minSizes     = layer->minSizes();
        auto maxSizes     = layer->maxSizes();
        auto aspectRatios = layer->aspectRatios();

        int flip         = layer->flip();
        int imageWidth   = layer->imageWidth();
        int imageHeight  = layer->imageHeight();
        float stepWidth  = layer->stepWidth();
        float stepHeight = layer->stepHeight();

        int imageW = imageWidth;
        int imageH = imageHeight;
        if (imageW <= 0) {
            imageW = inputTensor1->width();
        }
        if (imageH <= 0) {
            imageH = inputTensor1->height();
        }

        float stepW = stepWidth;
        float stepH = stepHeight;
        if (stepW <= 0) {
            stepW = (float)imageW / w;
        }

        if (stepH <= 0) {
            stepH = (float)imageH / h;
        }

        int minSizeCount = minSizes ? (int)minSizes->size() : 0;
        int maxSizeCount = maxSizes ? (int)maxSizes->size() : 0;
        std::vector<float> aspectRatiosValue{1.0f};
        if (aspectRatios != nullptr) {
            for (int i = 0; i < aspectRatios->size(); ++i) {
                auto ratio = aspectRatios->data()[i];
                bool exist = false;
                for (auto v : aspectRatiosValue) {
                    auto diff = v - ratio;
                    if (diff < 0) {
                        diff = -diff;
                    }
                    if (diff < 1e-6) {
                        exist = true;
                        break;
                    }
                }
                if (!exist) {
                    aspectRatiosValue.emplace_back(ratio);
                    if (flip) {
                        aspectRatiosValue.emplace_back(1.0f / ratio);
                    }
                }
            }
        }
        int priorCount = minSizeCount * aspectRatiosValue.size() + maxSizeCount;

        auto& outputTensorBuffer                              = outputs[0]->buffer();
        outputTensorBuffer.dim[0].extent                      = 1;
        outputTensorBuffer.dim[1].extent                      = 2;
        outputTensorBuffer.dim[2].extent                      = 4 * w * h * priorCount;
        outputTensorBuffer.dim[3].extent                      = 1;
        outputTensorBuffer.type = halide_type_of<float>();
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;

        return true;
    }
};

REGISTER_SHAPE(PriorBoxComputer, OpType_PriorBox);
} // namespace MNN
