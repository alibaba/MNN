//
//  GeometryShape.cpp
//  MNN
//
//  Created by MNN on 2021/03/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include "core/AutoStorage.h"
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"

namespace MNN {
class GeometryShape : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        if(!context.allocTensor(outputs[0])) {
            return false;
        }
        auto& ib         = inputs[0]->buffer();
        auto outputData = outputs[0]->host<int>();
        auto inputFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        if ((inputFormat == MNN_DATA_FORMAT_NC4HW4) && TensorUtils::getDescribe(outputs[0])->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
            outputData[0] = ib.dim[0].extent;
            outputData[1] = ib.dim[2].extent;
            outputData[2] = ib.dim[3].extent;
            outputData[3] = ib.dim[1].extent;
        } else {
            for (int i = 0; i < ib.dimensions; i++) {
                outputData[i] = ib.dim[i].extent;
            }
        }
        return true;
    }
};

class GeometryRank : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        if(!context.allocTensor(outputs[0])) {
            return false;
        }
        outputs[0]->host<int>()[0] = inputs[0]->buffer().dimensions;
        return true;
    }
};

class GeometryPriorBox : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        if(!context.allocTensor(outputs[0])) {
            return false;
        }
        std::shared_ptr<Tensor> outputTemp(new Tensor(outputs[0], Tensor::CAFFE));
        if (nullptr == outputTemp->host<void>()) {
            // Out of memory
            return false;
        }
        auto layer  = op->main_as_PriorBox();
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
        auto minSizes     = layer->minSizes();
        auto minSizeCount = minSizes ? minSizes->size() : 0;
        auto maxSizes     = layer->maxSizes();
        auto maxSizeCount = maxSizes ? maxSizes->size() : 0;
        auto aspectRatios = layer->aspectRatios();
        bool flip         = layer->flip();

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

        // boxes
        float offset  = layer->offset();
        auto boxesPtr = outputTemp->host<float>();
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
                    for (int p = 0; p < aspectRatiosValue.size(); p++) {
                        float arsqrt = sqrt(aspectRatiosValue[p]);
                        if (fabsf(arsqrt - 1.0f) < 1e-6) {
                            continue;
                        }
                        float boxW = minSize * arsqrt;
                        float boxH = minSize / arsqrt;

                        box[0] = (centerX - boxW * 0.5f) / imageW;
                        box[1] = (centerY - boxH * 0.5f) / imageH;
                        box[2] = (centerX + boxW * 0.5f) / imageW;
                        box[3] = (centerY + boxH * 0.5f) / imageH;
                        box += 4;
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
        auto outputData = outputs[0]->host<float>();
        MNNCPUCopyBuffer(outputTemp.get(), outputs[0]);
        return true;
    }
};

class GeometrySize : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        if(!context.allocTensor(outputs[0])) {
            return false;
        }
        int count = 1;
        for (int i = 0; i < inputs[0]->buffer().dimensions; i++) {
            count *= inputs[0]->buffer().dim[i].extent;
        }
        outputs[0]->host<int>()[0] = count;
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryShape);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Shape});
    std::shared_ptr<GeometryComputer> comp1(new GeometryRank);
    GeometryComputer::registerGeometryComputer(comp1, {OpType_Rank});
    std::shared_ptr<GeometryComputer> comp2(new GeometryPriorBox);
    GeometryComputer::registerGeometryComputer(comp2, {OpType_PriorBox});
    std::shared_ptr<GeometryComputer> comp3(new GeometrySize);
    GeometryComputer::registerGeometryComputer(comp3, {OpType_Size});
}

REGISTER_GEOMETRY(GeometryShape, _create);

} // namespace MNN
