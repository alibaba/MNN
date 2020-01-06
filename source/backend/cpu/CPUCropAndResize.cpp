//
//  CPUCropAndResize.cpp
//  MNN
//
//  Created by MNN on 2018/08/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUCropAndResize.hpp"
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

template <typename T>
CPUCropAndResize<T>::CPUCropAndResize(Backend* backend, const Op* op) : Execution(backend) {
    auto cr             = op->main_as_CropAndResize();
    mMethod             = cr->method();
    mExtrapolationValue = cr->extrapolationValue();
}

template <typename T>
const ErrorCode CPUCropAndResize<T>::CropAndResize(const Tensor* image, const Tensor* boxes, const Tensor* boxIndex,
                                                   Tensor* crops) {
    const int batchSize   = image->buffer().dim[0].extent;
    const int imageHeight = image->buffer().dim[1].extent;
    const int imageWidth  = image->buffer().dim[2].extent;
    const int imageDepth  = image->buffer().dim[3].extent;

    MNN_ASSERT(imageWidth > 0 && imageHeight > 0);

    const int numBoxes   = crops->buffer().dim[0].extent;
    const int cropHeight = crops->buffer().dim[1].extent;
    const int cropWidth  = crops->buffer().dim[2].extent;
    const int depth      = crops->buffer().dim[3].extent;

    // init
    memset(crops->host<float>(), 0, crops->size());

    // Sharding across boxes.
    auto CropAndResizePerBox = [&](int startBox, int limitBox) {
        for (int b = startBox; b < limitBox; ++b) {
            const float y1 = boxes->host<float>()[b * 4];
            const float x1 = boxes->host<float>()[b * 4 + 1];
            const float y2 = boxes->host<float>()[b * 4 + 2];
            const float x2 = boxes->host<float>()[b * 4 + 3];

            const int32_t bIn = boxIndex->host<int32_t>()[b];
            if (0 > bIn || bIn >= batchSize) {
                continue;
            }

            const float heightScale = (cropHeight > 1) ? (y2 - y1) * (imageHeight - 1) / (cropHeight - 1) : 0;
            const float widthScale  = (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / (cropWidth - 1) : 0;

            int32_t cropsHeight = crops->buffer().dim[1].extent;
            int32_t cropsWidth  = crops->buffer().dim[2].extent;
            int32_t cropsDepth  = crops->buffer().dim[3].extent;

            for (int y = 0; y < cropHeight; ++y) {
                const float inY =
                    (cropHeight > 1) ? y1 * (imageHeight - 1) + y * heightScale : 0.5 * (y1 + y2) * (imageHeight - 1);
                if (inY < 0 || inY > imageHeight - 1) {
                    for (int x = 0; x < cropWidth; ++x) {
                        for (int d = 0; d < depth; ++d) {
                            crops->host<float>()[b * cropsHeight * cropsWidth * cropsDepth +
                                                 y * cropsWidth * cropsDepth + x * cropsDepth + d] =
                                mExtrapolationValue;
                        }
                    }
                    continue;
                }
                if (mMethod == CropAndResizeMethod_BILINEAR) {
                    const int topYIndex    = floorf(inY);
                    const int bottomYIndex = ceilf(inY);
                    const float yLerp      = inY - topYIndex;

                    for (int x = 0; x < cropWidth; ++x) {
                        const float inX = (cropWidth > 1) ? x1 * (imageWidth - 1) + x * widthScale
                                                          : 0.5 * (x1 + x2) * (imageWidth - 1);
                        if (inX < 0 || inX > imageWidth - 1) {
                            for (int d = 0; d < depth; ++d) {
                                crops->host<float>()[b * cropsHeight * cropsWidth * cropsDepth +
                                                     y * cropsWidth * cropsDepth + x * cropsDepth + d] =
                                    mExtrapolationValue;
                            }
                            continue;
                        }
                        const int leftXIndex  = floorf(inX);
                        const int rightXIndex = ceilf(inX);
                        const float xLerp     = inX - leftXIndex;

                        for (int d = 0; d < depth; ++d) {
                            const float topLeft(
                                static_cast<float>(image->host<float>()[bIn * imageHeight * imageWidth * imageDepth +
                                                                        topYIndex * imageWidth * imageDepth +
                                                                        leftXIndex * imageDepth + d]));
                            const float topRight(
                                static_cast<float>(image->host<float>()[bIn * imageHeight * imageWidth * imageDepth +
                                                                        topYIndex * imageWidth * imageDepth +
                                                                        rightXIndex * imageDepth + d]));
                            const float bottomLeft(
                                static_cast<float>(image->host<float>()[bIn * imageHeight * imageWidth * imageDepth +
                                                                        bottomYIndex * imageWidth * imageDepth +
                                                                        leftXIndex * imageDepth + d]));
                            const float bottomRight(
                                static_cast<float>(image->host<float>()[bIn * imageHeight * imageWidth * imageDepth +
                                                                        bottomYIndex * imageWidth * imageDepth +
                                                                        rightXIndex * imageDepth + d]));

                            const float top    = topLeft + (topRight - topLeft) * xLerp;
                            const float bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
                            crops->host<float>()[b * cropsHeight * cropsWidth * cropsDepth +
                                                 y * cropsWidth * cropsDepth + x * cropsDepth + d] =
                                top + (bottom - top) * yLerp;
                        }
                    }
                } else if (mMethod == CropAndResizeMethod_NEAREST) { // method == "nearest"
                    for (int x = 0; x < cropWidth; ++x) {
                        const float inX = (cropWidth > 1) ? x1 * (imageWidth - 1) + x * widthScale
                                                          : 0.5 * (x1 + x2) * (imageWidth - 1);
                        if (inX < 0 || inX > imageWidth - 1) {
                            for (int d = 0; d < depth; ++d) {
                                crops->host<float>()[b * cropsHeight * cropsWidth * cropsDepth +
                                                     y * cropsWidth * cropsDepth + x * cropsDepth + d] =
                                    mExtrapolationValue;
                            }
                            continue;
                        }
                        const int closestXIndex = roundf(inX);
                        const int closestYIndex = roundf(inY);
                        for (int d = 0; d < depth; ++d) {
                            crops->host<float>()[b * cropsHeight * cropsWidth * cropsDepth +
                                                 y * cropsWidth * cropsDepth + x * cropsDepth + d] =
                                static_cast<float>(image->host<float>()[bIn * imageHeight * imageWidth * imageDepth +
                                                                        closestYIndex * imageWidth * imageDepth +
                                                                        closestXIndex * imageDepth + d]);
                        }
                    }
                } else {
                    MNN_ASSERT(false);
                }
            }
        }
    };

    for (int i = 0; i < numBoxes; i++) {
        CropAndResizePerBox(i, i + 1);
    }
    return NO_ERROR;
}

template <typename T>
ErrorCode CPUCropAndResize<T>::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // The shape of 'image' is [batch_size, image_height, image_width,
    // channels].
    const Tensor* image = inputs[0];
    // The shape of 'boxes' is [num_boxes, 4].
    const Tensor* boxes = inputs[1];
    // The shape of 'box_index' is [num_boxes].
    const Tensor* boxIndex = inputs[2];

    const ErrorCode status = CropAndResize(image, boxes, boxIndex, outputs[0]);
    return status;
}

class CPUCropAndResizeCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUCropAndResize<int32_t>(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUCropAndResizeCreator, OpType_CropAndResize);
} // namespace MNN
