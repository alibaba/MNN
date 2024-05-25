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

namespace MNN {

CPUROIAlign::CPUROIAlign(Backend* backend, int pooledWidth, int pooledHeight, int samplingRatio, float spatialScale,
                         bool aligned, PoolType poolType, bool outputGrad)
    : Execution(backend),
      mPooledWidth(pooledWidth),
      mPooledHeight(pooledHeight),
      mSamplingRatio(samplingRatio),
      mSpatialScale(spatialScale),
      mAligned(aligned),
      mPoolType(poolType),
      mOutputGrad(outputGrad) {
    // nothing to do
}

ErrorCode CPUROIAlign::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    //
    auto& roi = inputs[1]->buffer();

    mROI.buffer().dimensions = roi.dimensions;
    mROI.buffer().type = halide_type_of<float>();
    memcpy(mROI.buffer().dim, roi.dim, sizeof(halide_dimension_t) * roi.dimensions);
    TensorUtils::getDescribe(&mROI)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
    TensorUtils::setLinearLayout(&mROI);
    auto core    = static_cast<CPUBackend*>(backend())->functions();
    if (core->bytes < 4) {
        mROITemp.reset(MNN::Tensor::createDevice<int32_t>({mROI.elementSize()}));
    }
    auto res = backend()->onAcquireBuffer(&mROI, Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    if (core->bytes < 4) {
        res = backend()->onAcquireBuffer(mROITemp.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
    }
    backend()->onReleaseBuffer(&mROI, Backend::DYNAMIC);
    if (core->bytes < 4) {
        backend()->onReleaseBuffer(mROITemp.get(), Backend::DYNAMIC);
    }

    return NO_ERROR;
}

ErrorCode CPUROIAlign::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& input  = inputs[0];
    auto& output = outputs[0];
    auto core    = static_cast<CPUBackend*>(backend())->functions();

    CPUTensorConverter::convert(inputs[1], &mROI, core);

    // dataType of ROI must be float32.
    Tensor *roiTensor = &mROI;
    auto roiPtrSrc = roiTensor->host<float>();
    if (core->bytes != 4) {
        core->MNNLowpToFp32(mROI.host<int16_t>(), mROITemp->host<float>(), mROI.elementSize());
        roiPtrSrc = mROITemp->host<float>();
    }

    if (mOutputGrad == false) {
        // get params
        auto iw = input->width(), ih = input->height(), is = iw * ih * core->pack;   // C4
        auto ow = output->width(), oh = output->height(), os = ow * oh * core->pack; // C4
        auto rs           = roiTensor->stride(0);
        auto numROI       = roiTensor->batch();
        auto numSlice     = UP_DIV(input->channel(), core->pack);
        float alignOffset = mAligned ? -0.5f : 0.f;

        for (int n = 0; n < numROI; ++n) {
            auto batchOutput = output->host<uint8_t>() + os * n * core->bytes;
            auto roiPtr      = roiPtrSrc + rs * n;
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
    } else {
        // get params
        auto iw = input->width(), ih = input->height(), is = iw * ih * core->pack;   // C4
        // backward mode, output shape is the same with input[0] shape
        // inputs[3] is backward diff
        auto ow = inputs[3]->width(), oh = inputs[3]->height(), os = ow * oh * core->pack; // C4
        auto rs           = roiTensor->stride(0);
        auto numROI       = roiTensor->batch();
        auto numSlice     = UP_DIV(input->channel(), core->pack);
        float alignOffset = mAligned ? -0.5f : 0.f;
        auto& bwDiff = inputs[3];
        ::memset(output->host<uint8_t>(), 0, static_cast<CPUBackend*>(backend())->getTensorSize(output, true));

        for (int n = 0; n < numROI; ++n) {
            auto roiPtr      = roiPtrSrc + rs * n;
            int batchIdx     = inputs[2]->host<int>()[n], idxRoi = 0;
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
            auto batchOutput = output->host<uint8_t>() + is * batchIdx * core->bytes;
            auto batchBwDiff = bwDiff->host<uint8_t>() + os * n * core->bytes;
            if (mPoolType == PoolType_AVEPOOL) {
                for (int s = 0; s < numSlice; ++s) {
                    auto sliceInput = (float*)(batchInput + is * input->batch() * s * core->bytes);
                    auto sliceOutput  = (float*)(batchOutput + is * input->batch() * s * core->bytes);
                    auto rowBwDiff  = (float*)(batchBwDiff + os * bwDiff->batch() * s * core->bytes);

                    int samplingRatioArea = samplingRatioH * samplingRatioW;
                    float invSamplingCnt = 1.f / samplingRatioArea;
                    for (int h = 0; h < mPooledHeight; ++h, rowBwDiff += mPooledWidth * core->pack) {
                        int preCalcIdx = h * mPooledWidth * samplingRatioArea;
                        for (int w = 0; w < mPooledWidth; ++w) {
                            float* localDiff = rowBwDiff + w * core->pack;
                            for (int i = 0; i < samplingRatioArea; ++i) {
                                const std::vector<int>& pos    = vecPos[preCalcIdx];
                                const std::vector<float>& area = vecArea[preCalcIdx];

                                for (int k = 0; k < core->pack; k++) {
                                    float dav = localDiff[k] * invSamplingCnt;
                                    (sliceOutput + pos[0] * core->pack)[k] += (dav * area[0]);
                                    (sliceOutput + pos[1] * core->pack)[k] += (dav * area[1]);
                                    (sliceOutput + pos[2] * core->pack)[k] += (dav * area[2]);
                                    (sliceOutput + pos[3] * core->pack)[k] += (dav * area[3]);
                                }
                                preCalcIdx++;
                            }
                        }
                    }
                }
            } else if (mPoolType == PoolType_MAXPOOL) {
                // TODO: the grad is not align with mmcv's result, but i don't find the bug
                for (int s = 0; s < numSlice; ++s) {
                    auto sliceInput = (float*)(batchInput + is * input->batch() * s * core->bytes);
                    auto sliceOutput  = (float*)(batchOutput + is * input->batch() * s * core->bytes);
                    auto rowBwDiff  = (float*)(batchBwDiff + os * bwDiff->batch() * s * core->bytes);

                    int samplingRatioArea = samplingRatioH * samplingRatioW;
                    for (int h = 0; h < mPooledHeight; ++h, rowBwDiff += mPooledWidth * core->pack) {
                        int preCalcIdx = h * mPooledWidth * samplingRatioArea;
                        for (int w = 0; w < mPooledWidth; ++w) {
                            float* localDiff = rowBwDiff + w * core->pack;

                            std::vector<float> maxVals(core->pack, -FLT_MAX);
                            std::vector<int> preCalcIdxVec(core->pack, 0);
                            for (int i = 0; i < samplingRatioArea; ++i) {
                                const std::vector<int>& pos    = vecPos[preCalcIdx];
                                const std::vector<float>& area = vecArea[preCalcIdx];

                                for (int k = 0; k < core->pack; k++) {
                                    float val0 = (sliceInput + pos[0] * core->pack)[k] * area[0];
                                    float val1 = (sliceInput + pos[1] * core->pack)[k] * area[1];
                                    float val2 = (sliceInput + pos[2] * core->pack)[k] * area[2];
                                    float val3 = (sliceInput + pos[3] * core->pack)[k] * area[3];
                                    float val = val0 + val1 + val2 + val3;
                                    if (val > maxVals[k]) {
                                        maxVals[k] = val;
                                        preCalcIdxVec[k] = preCalcIdx;
                                    }
                                }
                                preCalcIdx++;
                            }

                            for (int k = 0; k < core->pack; k++) {
                                const std::vector<int>& pos    = vecPos[preCalcIdxVec[k]];
                                const std::vector<float>& area = vecArea[preCalcIdxVec[k]];

                                (sliceOutput + pos[0] * core->pack)[k] += (localDiff[k] * area[0]);
                                (sliceOutput + pos[1] * core->pack)[k] += (localDiff[k] * area[1]);
                                (sliceOutput + pos[2] * core->pack)[k] += (localDiff[k] * area[2]);
                                (sliceOutput + pos[3] * core->pack)[k] += (localDiff[k] * area[3]);
                            }
                        }
                    }
                }
            } else {
                MNN_ERROR("grad of pooling mode: %d not supported now!", mPoolType);
                return NOT_SUPPORT;
            }
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
        if (core->bytes < 4 && roiAlign->outputGrad()) {
            return nullptr;
        }
        return new CPUROIAlign(backend, roiAlign->pooledWidth(), roiAlign->pooledHeight(), roiAlign->samplingRatio(),
                               roiAlign->spatialScale(), roiAlign->aligned(), roiAlign->poolType(), roiAlign->outputGrad());
    }
};

REGISTER_CPU_OP_CREATOR(CPUROIAlignCreator, OpType_ROIAlign);
} // namespace MNN
