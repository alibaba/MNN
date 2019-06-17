//
//  ShapePriorbox.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

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

        int minSizeCount     = minSizes ? minSizes->size() : 0;
        int maxSizeCount     = maxSizes ? maxSizes->size() : 0;
        int aspectRatioCount = aspectRatios ? aspectRatios->size() : 0;

        int priorCount = minSizeCount * aspectRatioCount + minSizeCount + maxSizeCount;
        if (flip) {
            priorCount += minSizeCount * aspectRatioCount;
        }

        auto& outputTensorBuffer         = outputs[0]->buffer();
        outputTensorBuffer.dim[0].extent = 1;
        outputTensorBuffer.dim[1].extent = 2;
        outputTensorBuffer.dim[2].extent = 4 * w * h * priorCount;
        outputTensorBuffer.dim[3].extent = 1;

        return true;
    }
};

REGISTER_SHAPE(PriorBoxComputer, OpType_PriorBox);
} // namespace MNN
