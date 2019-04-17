//
//  CPUPriorbox.cpp
//  MNN
//
//  Created by MNN on 2018/07/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUPriorbox.hpp"
#include <math.h>
#include "AutoStorage.h"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "TensorUtils.hpp"

namespace MNN {

CPUPriorBox::CPUPriorBox(Backend *b, const MNN::Op *op) : MNN::Execution(b) {
    mParameter = op->main_as_PriorBox();
}

ErrorCode CPUPriorBox::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}
ErrorCode CPUPriorBox::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    AutoStorage<float> mOutputData;
    mOutputData.reset(outputs[0]->height() * outputs[0]->channel());

    auto layer  = mParameter;
    auto input0 = inputs[0];
    const int w = input0->width();
    const int h = input0->height();

    // image width, height
    int imageW = layer->imageWidth();
    if (imageW <= 0) {
        imageW = inputs[1]->width();
    }
    int imageH = layer->imageHeight();
    if (imageH <= 0) {
        imageH = inputs[1]->height();
    }

    // step width, height
    float stepW = layer->stepWidth();
    if (stepW <= 0) {
        stepW = (float)imageW / w;
    }
    float stepH = layer->stepHeight();
    if (stepH <= 0) {
        stepH = (float)imageH / h;
    }

    // sizes
    auto minSizes         = layer->minSizes();
    auto minSizeCount     = minSizes ? minSizes->size() : 0;
    auto maxSizes         = layer->maxSizes();
    auto maxSizeCount     = maxSizes ? maxSizes->size() : 0;
    auto aspectRatios     = layer->aspectRatios();
    auto aspectRatioCount = aspectRatios ? aspectRatios->size() : 0;
    bool flip             = layer->flip();
    auto priorCount       = minSizeCount * aspectRatioCount + minSizeCount + maxSizeCount;
    if (flip) {
        priorCount += minSizeCount * aspectRatioCount;
    }

    // boxes
    float offset  = layer->offset();
    auto boxesPtr = mOutputData.get();
    for (int i = 0; i < h; i++) {
        float *box    = boxesPtr + i * w * priorCount * 4;
        float centerX = offset * stepW;
        float centerY = offset * stepH + i * stepH;
        for (int j = 0; j < w; j++, centerX += stepW) {
            for (int k = 0; k < minSizeCount; k++) {
                // min size box
                float minSize = minSizes->data()[k];
                {
                    box[0] = (centerX - minSize * 0.5f) / imageW;
                    box[1] = (centerY - minSize * 0.5f) / imageH;
                    box[2] = (centerX + minSize * 0.5f) / imageW;
                    box[3] = (centerY + minSize * 0.5f) / imageH;
                    box += 4;
                }

                // max size box
                if (maxSizeCount > 0) {
                    float maxSize = maxSizes->data()[k];
                    float ssqrt   = sqrt(minSize * maxSize);

                    box[0] = (centerX - ssqrt * 0.5f) / imageW;
                    box[1] = (centerY - ssqrt * 0.5f) / imageH;
                    box[2] = (centerX + ssqrt * 0.5f) / imageW;
                    box[3] = (centerY + ssqrt * 0.5f) / imageH;
                    box += 4;
                }

                // aspect ratios
                for (int p = 0; p < aspectRatioCount; p++) {
                    float arsqrt = sqrt(aspectRatios->data()[p]);
                    float boxW   = minSize * arsqrt;
                    float boxH   = minSize / arsqrt;

                    box[0] = (centerX - boxW * 0.5f) / imageW;
                    box[1] = (centerY - boxH * 0.5f) / imageH;
                    box[2] = (centerX + boxW * 0.5f) / imageW;
                    box[3] = (centerY + boxH * 0.5f) / imageH;
                    box += 4;

                    if (flip) {
                        box[0] = (centerX - boxH * 0.5f) / imageH;
                        box[1] = (centerY - boxW * 0.5f) / imageW;
                        box[2] = (centerX + boxH * 0.5f) / imageH;
                        box[3] = (centerY + boxW * 0.5f) / imageW;
                        box += 4;
                    }
                }
            }
        }
    }

    // clip
    int oh = outputs[0]->height();
    if (layer->clip()) {
        float *box = boxesPtr;
        for (int i = 0; i < oh; i++) {
            box[i] = std::min(std::max(box[i], 0.f), 1.f);
        }
    }

    // set variance
    auto variances = layer->variances()->data();
    auto var       = boxesPtr + oh;
    for (int i = 0; i < oh / 4; i++) {
        var[0] = variances[0];
        var[1] = variances[1];
        var[2] = variances[2];
        var[3] = variances[3];
        var += 4;
    }

    // transform to output
    auto output = outputs[0];
    MNNPackC4(output->host<float>(), mOutputData.get(), output->height(), output->channel());
    return NO_ERROR;
}

class CPUPriorBoxCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUPriorBox(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUPriorBoxCreator, OpType_PriorBox);
} // namespace MNN
