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
    auto rs           = mROI.stride(0);
    auto numROI       = mROI.batch();
    auto numSlice     = UP_DIV(input->channel(), 4);
    float alignOffset = mAligned ? -0.5f : 0.f;

#ifndef MNN_USE_NEON
    memset(output->host<void>(), 0, output->elementSize() * sizeof(float));
#endif

    for (int n = 0; n < numROI; ++n) {
        auto batchOutput = output->host<float>() + os * n;
        auto roiPtr      = mROI.host<float>() + rs * n;
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

        float invSamplingCnt = 1.f / (samplingRatioH * samplingRatioW);

        std::vector<std::vector<int>> vecPos;
        std::vector<std::vector<float>> vecArea;
        preCalcBilinearInterpolate(ih, iw, mPooledHeight, mPooledWidth, y1, x1, binSizeH, binSizeW, samplingRatioH,
                                   samplingRatioW, vecPos, vecArea);

        auto batchInput = input->host<float>() + is * batchIdx;
        if (mPoolType == PoolType_AVEPOOL) {
            for (int s = 0; s < numSlice; ++s) {
                auto sliceInput = batchInput + is * input->batch() * s;
                auto rowOutput  = batchOutput + os * output->batch() * s;
                int preCalcIdx  = 0;
                for (int h = 0; h < mPooledHeight; ++h, rowOutput += mPooledWidth * 4) {
                    for (int w = 0; w < mPooledWidth; ++w) {
#ifdef MNN_USE_NEON
                        float32x4_t res = vmovq_n_f32(0.f);
                        for (int i = 0; i < samplingRatioH; ++i) {
                            for (int j = 0; j < samplingRatioW; ++j) {
                                std::vector<int>& pos    = vecPos[preCalcIdx];
                                std::vector<float>& area = vecArea[preCalcIdx];

                                float32x4_t val0 = vld1q_f32(sliceInput + pos[0]);
                                float32x4_t val1 = vld1q_f32(sliceInput + pos[1]);
                                float32x4_t val2 = vld1q_f32(sliceInput + pos[2]);
                                float32x4_t val3 = vld1q_f32(sliceInput + pos[3]);
                                float32x4_t mla  = vmulq_n_f32(val0, area[0]);
                                mla              = vmlaq_n_f32(mla, val1, area[1]);
                                mla              = vmlaq_n_f32(mla, val2, area[2]);
                                mla              = vmlaq_n_f32(mla, val3, area[3]);
                                res              = vaddq_f32(res, mla);
                                preCalcIdx++;
                            }
                        }
                        res = vmulq_n_f32(res, invSamplingCnt);
                        vst1q_f32(rowOutput + w * 4, res);
#elif defined(MNN_USE_SSE)
                        auto res = _mm_set_ps1(0.f);
                        for (int i = 0; i < samplingRatioH; ++i) {
                            for (int j = 0; j < samplingRatioW; ++j) {
                                std::vector<int>& pos    = vecPos[preCalcIdx];
                                std::vector<float>& area = vecArea[preCalcIdx];

                                auto val0 = _mm_loadu_ps(sliceInput + pos[0]);
                                auto val1 = _mm_loadu_ps(sliceInput + pos[1]);
                                auto val2 = _mm_loadu_ps(sliceInput + pos[2]);
                                auto val3 = _mm_loadu_ps(sliceInput + pos[3]);
                                auto mla  = _mm_mul_ps(val0, _mm_set_ps1(area[0]));
                                mla       = _mm_add_ps(_mm_mul_ps(val1, _mm_set_ps1(area[1])), mla);
                                mla       = _mm_add_ps(_mm_mul_ps(val2, _mm_set_ps1(area[2])), mla);
                                mla       = _mm_add_ps(_mm_mul_ps(val3, _mm_set_ps1(area[3])), mla);
                                res       = _mm_add_ps(res, mla);
                                preCalcIdx++;
                            }
                        }
                        res      = _mm_mul_ps(res, _mm_set_ps1(invSamplingCnt));
                        _mm_storeu_ps(rowOutput + w * 4, res);
#else
                        for (int i = 0; i < samplingRatioH; ++i) {
                            for (int j = 0; j < samplingRatioW; ++j) {
                                std::vector<int>& pos    = vecPos[preCalcIdx];
                                std::vector<float>& area = vecArea[preCalcIdx];
                                for (int k = 0; k < 4; ++k) {
                                    float val0 = *(sliceInput + pos[0] + k);
                                    float val1 = *(sliceInput + pos[1] + k);
                                    float val2 = *(sliceInput + pos[2] + k);
                                    float val3 = *(sliceInput + pos[3] + k);
                                    rowOutput[w * 4 + k] +=
                                        (val0 * area[0] + val1 * area[1] + val2 * area[2] + val3 * area[3]) *
                                        invSamplingCnt;
                                }
                                preCalcIdx++;
                            }
                        }
#endif
                    }
                }
            }
        } else if (mPoolType == PoolType_MAXPOOL) {
            for (int s = 0; s < numSlice; ++s) {
                auto sliceInput = batchInput + is * input->batch() * s;
                auto rowOutput  = batchOutput + os * output->batch() * s;
                int preCalcIdx  = 0;
                for (int h = 0; h < mPooledHeight; ++h, rowOutput += mPooledWidth * 4) {
                    for (int w = 0; w < mPooledWidth; ++w) {
#ifdef MNN_USE_NEON
                        float32x4_t res = vmovq_n_f32(-FLT_MAX);
                        for (int i = 0; i < samplingRatioH; ++i) {
                            for (int j = 0; j < samplingRatioW; ++j) {
                                std::vector<int>& pos    = vecPos[preCalcIdx];
                                std::vector<float>& area = vecArea[preCalcIdx];

                                float32x4_t val0 = vld1q_f32(sliceInput + pos[0]);
                                float32x4_t val1 = vld1q_f32(sliceInput + pos[1]);
                                float32x4_t val2 = vld1q_f32(sliceInput + pos[2]);
                                float32x4_t val3 = vld1q_f32(sliceInput + pos[3]);
                                float32x4_t mla  = vmulq_n_f32(val0, area[0]);
                                mla              = vmlaq_n_f32(mla, val1, area[1]);
                                mla              = vmlaq_n_f32(mla, val2, area[2]);
                                mla              = vmlaq_n_f32(mla, val3, area[3]);
                                res              = vmaxq_f32(res, mla);
                                preCalcIdx++;
                            }
                        }
                        vst1q_f32(rowOutput + w * 4, res);
#elif defined(MNN_USE_SSE)
                        auto res = _mm_set_ps1(-FLT_MAX);
                        for (int i = 0; i < samplingRatioH; ++i) {
                            for (int j = 0; j < samplingRatioW; ++j) {
                                std::vector<int>& pos    = vecPos[preCalcIdx];
                                std::vector<float>& area = vecArea[preCalcIdx];

                                auto val0  = _mm_loadu_ps(sliceInput + pos[0]);
                                auto val1  = _mm_loadu_ps(sliceInput + pos[1]);
                                auto val2  = _mm_loadu_ps(sliceInput + pos[2]);
                                auto val3  = _mm_loadu_ps(sliceInput + pos[3]);
                                auto mla   = _mm_mul_ps(val0, _mm_set_ps1(area[0]));
                                mla       = _mm_add_ps(_mm_mul_ps(val1, _mm_set_ps1(area[1])), mla);
                                mla       = _mm_add_ps(_mm_mul_ps(val2, _mm_set_ps1(area[2])), mla);
                                mla       = _mm_add_ps(_mm_mul_ps(val3, _mm_set_ps1(area[3])), mla);
                                res        = _mm_max_ps(res, mla);
                                preCalcIdx++;
                            }
                        }
                        _mm_storeu_ps(rowOutput + w * 4, res);
#else
                        std::vector<float> vecVal[4];
                        for (int i = 0; i < samplingRatioH; ++i) {
                            for (int j = 0; j < samplingRatioW; ++j) {
                                std::vector<int>& pos    = vecPos[preCalcIdx];
                                std::vector<float>& area = vecArea[preCalcIdx];
                                for (int k = 0; k < 4; ++k) {
                                    float val0 = *(sliceInput + pos[0] + k);
                                    float val1 = *(sliceInput + pos[1] + k);
                                    float val2 = *(sliceInput + pos[2] + k);
                                    float val3 = *(sliceInput + pos[3] + k);
                                    vecVal[k].emplace_back(val0 * area[0] + val1 * area[1] + val2 * area[2] +
                                                           val3 * area[3]);
                                }
                                preCalcIdx++;
                            }
                        }
                        for (int k = 0; k < 4; ++k) {
                            rowOutput[w * 4 + k] = *std::max_element(vecVal[k].begin(), vecVal[k].end());
                        }
#endif
                    }
                }
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
                    int pos0 = (py0 * width + px0) * 4, pos1 = (py0 * width + px1) * 4, pos2 = (py1 * width + px0) * 4,
                        pos3 = (py1 * width + px1) * 4;
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
        return new CPUROIAlign(backend, roiAlign->pooledWidth(), roiAlign->pooledHeight(), roiAlign->samplingRatio(),
                               roiAlign->spatialScale(), roiAlign->aligned(), roiAlign->poolType());
    }
};

REGISTER_CPU_OP_CREATOR(CPUROIAlignCreator, OpType_ROIAlign);
} // namespace MNN
