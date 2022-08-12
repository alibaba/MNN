//
//  CPUROIAlign.cpp
//  MNN
//
//  Created by MNN on 2021/11/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUROIAlign.hpp"
#include <float.h>
#include <math.h>
#include <algorithm>
#include "CPUTensorConvert.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/TensorUtils.hpp"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#ifdef MNN_USE_SSE
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

namespace MNN {

CPUROIAlign::CPUROIAlign(Backend* backend, int pooledWidth, int pooledHeight, int samplingRatio, float spatialScale,
                         bool aligned, PoolType poolType)
    : Execution(backend),
      mPooledWidth(pooledWidth),
      mPooledHeight(pooledHeight),
      mSamplingRatio(samplingRatio),
      mSpatialScale(spatialScale),
      mAligned(aligned),
      mPoolType(poolType) {
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
    auto core    = static_cast<CPUBackend*>(backend())->functions();

    CPUTensorConverter::convert(inputs[1], &mROI, core);

    // dataType of ROI must be float32.
    Tensor *roiTensor = &mROI;
    if (core->bytes != 4) {
        core->MNNLowpToFp32(mROI.host<int16_t>(), mROI.host<float>(), mROI.elementSize());
    }

    // get params
    auto iw = input->width(), ih = input->height(), is = iw * ih * core->pack;   // C4
    auto ow = output->width(), oh = output->height(), os = ow * oh * core->pack; // C4
    auto rs           = roiTensor->stride(0);
    auto numROI       = roiTensor->batch();
    auto numSlice     = UP_DIV(input->channel(), core->pack);
    float alignOffset = mAligned ? -0.5f : 0.f;

    for (int n = 0; n < numROI; ++n) {
        auto batchOutput = output->host<uint8_t>() + os * n * core->bytes;
        auto roiPtr      = roiTensor->host<float>() + rs * n;
        int batchIdx     = (int)roiPtr[0], idxRoi = 1;
        if (inputs.size() == 3) {
            batchIdx = inputs[2]->host<int>()[n];
            idxRoi = 0;
        }
        float x1         = roiPtr[idxRoi++] * mSpatialScale + alignOffset;
        float y1         = roiPtr[idxRoi++] * mSpatialScale + alignOffset;
        float x2         = roiPtr[idxRoi++] * mSpatialScale + alignOffset;
        float y2         = roiPtr[idxRoi++] * mSpatialScale + alignOffset;
        MNN_ASSERT(batchIdx < input->batch());

        float roiW = x2 - x1;
        float roiH = y2 - y1;
        if (!mAligned) {
            roiW = std::max(roiW, 1.f);
            roiH = std::max(roiH, 1.f);
        }

        float binSizeW = roiW / mPooledWidth;
        float binSizeH = roiH / mPooledHeight;

        int samplingRatioW = mSamplingRatio > 0 ? mSamplingRatio : static_cast<int>(ceilf(roiW / mPooledWidth));
        int samplingRatioH = mSamplingRatio > 0 ? mSamplingRatio : static_cast<int>(ceilf(roiH / mPooledHeight));
        MNN_ASSERT(samplingRatioH > 0 && samplingRatioW > 0);

        std::vector<std::vector<int>> vecPos;
        std::vector<std::vector<float>> vecArea;
        preCalcBilinearInterpolate(ih, iw, mPooledHeight, mPooledWidth, y1, x1, binSizeH, binSizeW, samplingRatioH,
                                   samplingRatioW, vecPos, vecArea);

        auto batchInput = input->host<uint8_t>() + is * batchIdx * core->bytes;
        if (mPoolType == PoolType_AVEPOOL) {
            for (int s = 0; s < numSlice; ++s) {
                auto sliceInput = batchInput + is * input->batch() * s * core->bytes;
                auto rowOutput  = batchOutput + os * output->batch() * s * core->bytes;
                core->MNNRoiAlignAvg((float *)rowOutput, (float *)sliceInput, vecPos, vecArea, samplingRatioH * samplingRatioW, mPooledHeight, mPooledWidth);
            }
        } else if (mPoolType == PoolType_MAXPOOL) {
            for (int s = 0; s < numSlice; ++s) {
                auto sliceInput = batchInput + is * input->batch() * s * core->bytes;
                auto rowOutput  = batchOutput + os * output->batch() * s * core->bytes;
                core->MNNRoiAlignMax((float *)rowOutput, (float *)sliceInput, vecPos, vecArea, samplingRatioH * samplingRatioW, mPooledHeight, mPooledWidth);
            }
        } else {
            MNN_ERROR("pooling mode: %d not supported now!", mPoolType);
            return NOT_SUPPORT;
        }
    }

    return NO_ERROR;
}

ErrorCode CPUROIAlign::preCalcBilinearInterpolate(int height, int width, int pooledHeight, int pooledWidth,
                                                  float roiStartH, float roiStartW, float binSizeH, float binSizeW,
                                                  int samplingRatioH, int samplingRatioW,
                                                  std::vector<std::vector<int>>& vecPos,
                                                  std::vector<std::vector<float>>& vecArea) {
    float samplingBinH = binSizeH / samplingRatioH;
    float samplingBinW = binSizeW / samplingRatioW;

    for (int h = 0; h < pooledHeight; ++h) {
        float samplingStartH = roiStartH + h * binSizeH;
        for (int w = 0; w < pooledWidth; ++w) {
            float samplingStartW = roiStartW + w * binSizeW;
            for (int i = 0; i < samplingRatioH; ++i) {
                float py = samplingStartH + (0.5 + i) * samplingBinH;
                for (int j = 0; j < samplingRatioW; ++j) {
                    float px = samplingStartW + (0.5 + j) * samplingBinW;
                    if (py < -1.f || py > height || px < -1.f || px > width) {
                        std::vector<int> pos({0, 0, 0, 0});
                        std::vector<float> area({0.f, 0.f, 0.f, 0.f});
                        vecPos.emplace_back(std::move(pos));
                        vecArea.emplace_back(std::move(area));
                        continue;
                    }
                    py = py < 0 ? 0 : py;
                    px = px < 0 ? 0 : px;

                    int py0 = static_cast<int>(py), px0 = static_cast<int>(px), py1, px1;
                    if (py0 >= height - 1) {
                        py1 = py0 = height - 1;
                        py        = static_cast<float>(py0);
                    } else {
                        py1 = py0 + 1;
                    }
                    if (px0 >= width - 1) {
                        px1 = px0 = width - 1;
                        px        = static_cast<float>(px0);
                    } else {
                        px1 = px0 + 1;
                    }

                    float dy0 = py - py0, dx0 = px - px0;
                    float dy1 = 1.f - dy0, dx1 = 1.f - dx0;
                    float area0 = dx0 * dy0, area1 = dx1 * dy0, area2 = dx0 * dy1, area3 = dx1 * dy1;
                    int pos0 = py0 * width + px0, pos1 = py0 * width + px1, pos2 = py1 * width + px0,
                        pos3 = py1 * width + px1;
                    std::vector<int> pos({pos0, pos1, pos2, pos3});
                    std::vector<float> area({area3, area2, area1, area0});
                    vecPos.emplace_back(std::move(pos));
                    vecArea.emplace_back(std::move(area));
                }
            }
        }
    }
    return NO_ERROR;
}

class CPUROIAlignCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto roiAlign = op->main_as_RoiParameters();
        auto core = static_cast<CPUBackend*>(backend)->functions();
        if (core->MNNRoiAlignMax == nullptr || core->MNNRoiAlignAvg == nullptr) {
            MNN_ERROR("Don't have function for CPUROIAlign\n");
            return nullptr;
        }
        return new CPUROIAlign(backend, roiAlign->pooledWidth(), roiAlign->pooledHeight(), roiAlign->samplingRatio(),
                               roiAlign->spatialScale(), roiAlign->aligned(), roiAlign->poolType());
    }
};

REGISTER_CPU_OP_CREATOR(CPUROIAlignCreator, OpType_ROIAlign);
} // namespace MNN
