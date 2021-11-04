//
//  CPUROIAlign.cpp
//  MNN
//
//  Created by MNN on 2021/11/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUROIAlign.hpp"
#include <algorithm>
#include "CPUTensorConvert.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/TensorUtils.hpp"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

CPUROIAlign::CPUROIAlign(Backend* backend, int pooledWidth, int pooledHeight, int samplingRatio, float spatialScale,
                         bool aligned, PoolMode poolMode)
    : Execution(backend),
      mPooledWidth(pooledWidth),
      mPooledHeight(pooledHeight),
      mSamplingRatio(samplingRatio),
      mSpatialScale(spatialScale),
      mAligned(aligned),
      mPoolMode(poolMode) {
    // nothing to do
}

ErrorCode CPUROIAlign::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    //
    auto& roi = inputs[1]->buffer();

    mROI.buffer().dimensions = roi.dimensions;
    memcpy(mROI.buffer().dim, roi.dim, sizeof(halide_dimension_t) * roi.dimensions);
    TensorUtils::getDescribe(&mROI)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
    TensorUtils::setLinearLayout(&mROI);

    backend()->onAcquireBuffer(&mROI, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mROI, Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode CPUROIAlign::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& input  = inputs[0];
    auto& output = outputs[0];

    // get roi
    MNN_DATA_FORMAT roiDimensionFormat = TensorUtils::getDescribe(inputs[1])->dimensionFormat;
    switch (roiDimensionFormat) {
        case MNN::MNN_DATA_FORMAT_NCHW:
            ::memcpy(mROI.host<void>(), inputs[1]->host<void>(), inputs[1]->elementSize() * sizeof(float));
            break;
        case MNN::MNN_DATA_FORMAT_NC4HW4:
            CPUTensorConverter::convert(inputs[1]->host<void>(), mROI.host<void>(), MNN_DATA_FORMAT_NC4HW4,
                                        MNN_DATA_FORMAT_NCHW, inputs[1]->batch(),
                                        inputs[1]->width() * inputs[1]->height(), inputs[1]->channel(),
                                        inputs[1]->getType().bytes(), static_cast<CPUBackend*>(backend())->functions());
            break;
        default:
            MNN_ERROR("rois dimension format: %d not supported now!", roiDimensionFormat);
            return NOT_SUPPORT;
            break;
    }

    // get params
    auto iw = input->width(), ih = input->height(), is = iw * ih * 4;   // C4
    auto ow = output->width(), oh = output->height(), os = ow * oh * 4; // C4
    auto rs       = mROI.stride(0);
    auto numROI   = mROI.batch();
    auto numSlice = UP_DIV(input->channel(), 4);

    memset(output->host<void>(), 0, output->elementSize() * sizeof(float));

    for (int n = 0; n < numROI; ++n) {
        auto batchOutput = output->host<float>() + os * n;
        auto roiPtr      = mROI.host<float>() + rs * n;
        int batchIdx     = static_cast<int>(roiPtr[0]);
        float x1         = roiPtr[1] * mSpatialScale;
        float y1         = roiPtr[2] * mSpatialScale;
        float x2         = roiPtr[3] * mSpatialScale;
        float y2         = roiPtr[4] * mSpatialScale;
        MNN_ASSERT(batchIdx < input->batch());

        float roiW = std::max((x2 - x1), 0.f);
        float roiH = std::max((y2 - y1), 0.f);

        float binSizeW = roiW / mPooledWidth;
        float binSizeH = roiH / mPooledHeight;

        int samplingRatioW   = mSamplingRatio > 0 ? mSamplingRatio : static_cast<int>(ceilf(roiW / mPooledWidth));
        int samplingRatioH   = mSamplingRatio > 0 ? mSamplingRatio : static_cast<int>(ceilf(roiH / mPooledHeight));
        float invSamplingCnt = 1.f / (samplingRatioH * samplingRatioW);

        float samplingBinW = binSizeW / samplingRatioW;
        float samplingBinH = binSizeH / samplingRatioH;

        float alignShift = mAligned ? -0.5f : 0.f;

        auto batchInput = input->host<float>() + is * batchIdx;
        if (mPoolMode == PoolMode_AvePool) {
            for (int s = 0; s < numSlice; ++s) {
                auto sliceInput = batchInput + is * input->batch() * s;
                auto rowOutput  = batchOutput + os * output->batch() * s;
                for (int h = 0; h < mPooledHeight; ++h, rowOutput += mPooledHeight * 4) {
                    float samplingStartH = y1 + h * binSizeH;
                    for (int w = 0; w < mPooledWidth; ++w) {
                        float samplingStartW = x1 + w * binSizeW;
                        for (int i = 0; i < samplingRatioH; ++i) {
                            float py  = std::max(samplingStartH + (0.5f + i) * samplingBinH + alignShift, 0.f);
                            int py0   = static_cast<int>(py);
                            int py1   = py0 + 1;
                            float dy0 = py - py0;
                            float dy1 = py1 - py;
                            for (int j = 0; j < samplingRatioW; ++j) {
                                float px    = std::max(samplingStartW + (0.5f + j) * samplingBinW + alignShift, 0.f);
                                int px0     = static_cast<int>(px);
                                int px1     = px0 + 1;
                                float dx0   = px - px0;
                                float dx1   = px1 - px;
                                float area0 = dx0 * dy0, area1 = dx1 * dy0, area2 = dx0 * dy1, area3 = dx1 * dy1;
                                for (int k = 0; k < 4; ++k) {
                                    float val0 = *(sliceInput + (py0 * iw + px0) * 4 + k);
                                    float val1 = *(sliceInput + (py0 * iw + px1) * 4 + k);
                                    float val2 = *(sliceInput + (py1 * iw + px0) * 4 + k);
                                    float val3 = *(sliceInput + (py1 * iw + px1) * 4 + k);
                                    rowOutput[w * 4 + k] +=
                                        (val0 * area3 + val1 * area2 + val2 * area1 + val3 * area0) * invSamplingCnt;
                                }
                            }
                        }
                    }
                }
            }
        } else if (mPoolMode == PoolMode_MaxPool) {
            for (int s = 0; s < numSlice; ++s) {
                auto sliceInput = batchInput + is * input->batch() * s;
                auto rowOutput  = batchOutput + os * output->batch() * s;
                for (int h = 0; h < mPooledHeight; ++h, rowOutput += mPooledHeight * 4) {
                    float samplingStartH = y1 + h * binSizeH;
                    for (int w = 0; w < mPooledWidth; ++w) {
                        float samplingStartW = x1 + w * binSizeW;
                        std::vector<float> vecVal[4];
                        for (int i = 0; i < samplingRatioH; ++i) {
                            float py  = std::max(samplingStartH + (0.5f + i) * samplingBinH + alignShift, 0.f);
                            int py0   = static_cast<int>(py);
                            int py1   = py0 + 1;
                            float dy0 = py - py0;
                            float dy1 = py1 - py;
                            for (int j = 0; j < samplingRatioW; ++j) {
                                float px    = std::max(samplingStartW + (0.5f + j) * samplingBinW + alignShift, 0.f);
                                int px0     = static_cast<int>(px);
                                int px1     = px0 + 1;
                                float dx0   = px - px0;
                                float dx1   = px1 - px;
                                float area0 = dx0 * dy0, area1 = dx1 * dy0, area2 = dx0 * dy1, area3 = dx1 * dy1;
                                for (int k = 0; k < 4; ++k) {
                                    float val0 = *(sliceInput + (py0 * iw + px0) * 4 + k);
                                    float val1 = *(sliceInput + (py0 * iw + px1) * 4 + k);
                                    float val2 = *(sliceInput + (py1 * iw + px0) * 4 + k);
                                    float val3 = *(sliceInput + (py1 * iw + px1) * 4 + k);
                                    vecVal[k].emplace_back(val0 * area3 + val1 * area2 + val2 * area1 + val3 * area0);
                                }
                            }
                        }
                        for (int k = 0; k < 4; ++k) {
                            rowOutput[w * 4 + k] = *std::max_element(vecVal[k].begin(), vecVal[k].end());
                        }
                    }
                }
            }
        } else {
            MNN_ERROR("pooling mode: %d not supported now!", mPoolMode);
            return NOT_SUPPORT;
        }
    }

    return NO_ERROR;
}

class CPUROIAlignCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto roiAlign = op->main_as_RoiAlign();
        return new CPUROIAlign(backend, roiAlign->pooledWidth(), roiAlign->pooledHeight(), roiAlign->samplingRatio(),
                               roiAlign->spatialScale(), roiAlign->aligned(), roiAlign->poolMode());
    }
};

REGISTER_CPU_OP_CREATOR(CPUROIAlignCreator, OpType_ROIAlign);
} // namespace MNN