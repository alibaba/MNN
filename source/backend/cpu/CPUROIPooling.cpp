//
//  CPUROIPooling.cpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUROIPooling.hpp"
#include <float.h>
#include <math.h>
#include "CPUTensorConvert.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

CPUROIPooling::CPUROIPooling(Backend *backend, int pooledWidth, int pooledHeight, float spatialScale, bool outputGrad)
    : CPUROIAlign(backend, pooledWidth, pooledHeight, 0, spatialScale, false, PoolType_MAX, outputGrad) {
    // nothing to do
}

static inline int max(int a, int b) { return a > b ? a : b; }
static inline int min(int a, int b) { return a < b ? a : b; }

ErrorCode CPUROIPooling::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto &input  = inputs[0];
    auto &output = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();

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
        auto iw = input->width(), ih = input->height(), is = iw * ih * core->pack;
        auto ow = output->width(), oh = output->height(), os = ow * oh * core->pack;
        auto slice  = UP_DIV(input->channel(), core->pack);
        auto numROI = inputs[1]->batch();
        for (int n = 0; n < numROI; ++n) {
            auto batchOutput = output->host<uint8_t>() + os * n * core->bytes;
            auto roiPtr      = roiPtrSrc + roiTensor->buffer().dim[0].stride * n;
            int roi          = roiPtr[0];
            int x1           = round(roiPtr[1] * mSpatialScale);
            int y1           = round(roiPtr[2] * mSpatialScale);
            int x2           = round(roiPtr[3] * mSpatialScale);
            int y2           = round(roiPtr[4] * mSpatialScale);
            MNN_ASSERT(roi < input->batch());

            int roiW = max(x2 - x1 + 1, 1);
            int roiH = max(y2 - y1 + 1, 1);

            float binSizeW = (float)roiW / (float)mPooledWidth;
            float binSizeH = (float)roiH / (float)mPooledHeight;

            auto batchInput = input->host<uint8_t>() + is * roi * core->bytes;
            for (int s = 0; s < slice; s++) {
                auto sliceInput = batchInput + is * input->batch() * s * core->bytes;
                auto rowOutput  = batchOutput + os * output->batch() * s * core->bytes;
                float binPh     = 0;
                for (int ph = 0; ph < mPooledHeight; ph++, rowOutput += mPooledWidth * core->pack * core->bytes) {
                    // Compute pooling region for this output unit:
                    //  start (included) = floor(ph * roiHeight / pooledHeight)
                    //  end (excluded) = ceil((ph + 1) * roiHeight / pooledHeight)
                    int hStart = min(max(y1 + (int)floorf(binPh), 0), ih);
                    binPh += binSizeH;
                    int hEnd = min(max(y1 + (int)ceilf(binPh), 0), ih);
                    int hLen = hEnd - hStart;
                    if (hLen <= 0) {
                        memset(rowOutput, 0, mPooledWidth * core->pack * core->bytes * sizeof(uint8_t));
                        continue;
                    }

                    float binPw = 0;
                    for (int pw = 0; pw < mPooledWidth; pw++) {
                        int wStart = min(max(x1 + (int)floorf(binPw), 0), iw);
                        binPw += binSizeW;
                        int wEnd = min(max(x1 + (int)ceilf(binPw), 0), iw);
                        int wLen = wEnd - wStart;
                        if (wLen <= 0) {
                            memset(rowOutput + pw * core->pack * core->bytes, 0, core->pack * core->bytes * sizeof(uint8_t));
                            continue;
                        }
                        core->MNNRoiPoolingMax((float *)(rowOutput + pw * core->pack * core->bytes), (float *)(sliceInput + (hStart * iw + wStart) * core->pack * core->bytes), hLen, wLen, iw);
                    }
                }
            }
        }
    } else {
        // get params
        auto iw = input->width(), ih = input->height(), is = iw * ih * core->pack;
        // backward mode, output shape is the same with input[0] shape
        auto& bwDiff = inputs[2];
        auto ow = bwDiff->width(), oh = bwDiff->height(), os = ow * oh * core->pack;
        auto slice  = UP_DIV(input->channel(), core->pack);
        auto numROI = inputs[1]->batch();
        ::memset(output->host<uint8_t>(), 0, static_cast<CPUBackend*>(backend())->getTensorSize(output, true));
        for (int n = 0; n < numROI; ++n) {
            auto batchBwDiff = inputs[2]->host<uint8_t>() + os * n * core->bytes;
            auto roiPtr      = roiPtrSrc + roiTensor->buffer().dim[0].stride * n;
            int roi          = roiPtr[0];
            int x1           = round(roiPtr[1] * mSpatialScale);
            int y1           = round(roiPtr[2] * mSpatialScale);
            int x2           = round(roiPtr[3] * mSpatialScale);
            int y2           = round(roiPtr[4] * mSpatialScale);
            MNN_ASSERT(roi < input->batch());

            int roiW = max(x2 - x1 + 1, 1);
            int roiH = max(y2 - y1 + 1, 1);

            float binSizeW = (float)roiW / (float)mPooledWidth;
            float binSizeH = (float)roiH / (float)mPooledHeight;

            auto batchInput = input->host<uint8_t>() + is * roi * core->bytes;
            auto batchOutput = output->host<uint8_t>() + is * roi * core->bytes;
            for (int s = 0; s < slice; s++) {
                auto sliceInput = batchInput + is * input->batch() * s * core->bytes;
                auto sliceOutput = batchOutput + is * input->batch() * s * core->bytes;
                auto rowBwDiff  = batchBwDiff + os * bwDiff->batch() * s * core->bytes;
                float binPh     = 0;
                for (int ph = 0; ph < mPooledHeight; ph++, rowBwDiff += mPooledWidth * core->pack * core->bytes) {
                    // Compute pooling region for this output unit:
                    //  start (included) = floor(ph * roiHeight / pooledHeight)
                    //  end (excluded) = ceil((ph + 1) * roiHeight / pooledHeight)
                    int hStart = min(max(y1 + (int)floorf(binPh), 0), ih);
                    binPh += binSizeH;
                    int hEnd = min(max(y1 + (int)ceilf(binPh), 0), ih);
                    int hLen = hEnd - hStart;
                    if (hLen <= 0) {
                        continue;
                    }

                    float binPw = 0;
                    for (int pw = 0; pw < mPooledWidth; pw++) {
                        int wStart = min(max(x1 + (int)floorf(binPw), 0), iw);
                        binPw += binSizeW;
                        int wEnd = min(max(x1 + (int)ceilf(binPw), 0), iw);
                        int wLen = wEnd - wStart;
                        if (wLen <= 0) {
                            continue;
                        }
                        {
                            std::vector<int> indices(core->pack);
                            std::vector<float> maxes(core->pack, -FLT_MAX);
                            float* src = (float *)(sliceInput + (hStart * iw + wStart) * core->pack * core->bytes);
                            float* diff = (float *)(rowBwDiff + pw * core->pack * core->bytes);
                            for (int h = 0; h < hLen; h++, src += iw * core->pack) {
                                for (int w = 0; w < wLen; w++) {
                                    int spatialIndex = (h + hStart) * iw + (wStart + w);
                                    float* srcPtr = src + w * core->pack;
                                    std::vector<float*> pre(core->pack, nullptr);
                                    for (int k = 0; k < core->pack; k++) {
                                        if (srcPtr[k] > maxes[k]) {
                                            maxes[k] = srcPtr[k];
                                            indices[k] = spatialIndex;
                                        }
                                    }
                                }
                            }
                            for (int k = 0; k < core->pack; k++) {
                                int h = indices[k] / iw;
                                int w = indices[k] % iw;
                                float* out = (float *)(sliceOutput + (h * iw + w) * core->pack * core->bytes);
                                out[k] += diff[k];
                            }
                        }
                    }
                }
            }
        }
    }

    return NO_ERROR;
}

class CPUROIPoolingCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto roi = op->main_as_RoiParameters();
        auto core = static_cast<CPUBackend*>(backend)->functions();
        if (core->MNNRoiPoolingMax == nullptr) {
            MNN_ERROR("Don't have function for CPUROIPooling\n");
            return nullptr;
        }
        if (core->bytes < 4 && roi->outputGrad()) {
            return nullptr;
        }
        return new CPUROIPooling(backend, roi->pooledWidth(), roi->pooledHeight(), roi->spatialScale(), roi->outputGrad());
    }
};
REGISTER_CPU_OP_CREATOR(CPUROIPoolingCreator, OpType_ROIPooling);

} // namespace MNN
